#!/usr/bin/env python3
"""
member_system_postgresql.py - ระบบจัดการสมาชิก PET Detect

- Flask Web Application สำหรับดูคะแนน สมาชิก และจัดการข้อมูล
- ใช้ PostgreSQL เท่านั้น (เช่น Render หรือเซิร์ฟเวอร์ Postgres)
- ตารางหลัก: scan_logs (ประวัติสแกน), admin_users (ผู้ดูแล), member_names (ชื่อสมาชิก)
- API สำหรับ Pi client: /api/add_score (บันทึกคะแนนจากการสแกน)
"""

import os
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')

# ==================== Database Config ====================
# โหลดจาก Environment (Render) หรือใช้ค่าเริ่มต้น
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'pet_detect_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'password')
}


print(f"Database Config: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']} (user: {DB_CONFIG['user']})")

# ==================== Database Connection & Init ====================
def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล PostgreSQL"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL")
        return connection
    
    except psycopg2.Error as e:
        print(f"PostgreSQL connection failed: {e}")
        return None


def init_database():
    """สร้างตารางถ้ายังไม่มี"""
    try:
        connection = get_db_connection()
        # ถ้าเชื่อมต่อเข้า Database ไม่ได้จะส่งค่าคืน
        if not connection:
            return False
        
        cursor = connection.cursor()
# สร้่างตาราง database
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_logs (
                id SERIAL PRIMARY KEY,
                rfid_id VARCHAR(50) NOT NULL,
                bottle_count INTEGER DEFAULT 0,
                can_count INTEGER DEFAULT 0,
                cap_count INTEGER DEFAULT 0,
                label_count INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0,
                image_path VARCHAR(500),
                scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS member_names (
                rfid_id VARCHAR(50) PRIMARY KEY,
                display_name VARCHAR(100) NOT NULL
            )
        """)
        
        connection.commit()
        cursor.close()
        connection.close()
        # หลังจากสร้างตารางเสร็จแล้ว ให้ส่ง Boolean True กลับไปเพ
        print("Database tables created successfully!")
        return True
        
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False

# ==================== Authentication Utilities ====================
def hash_password(password):
    """เข้ารหัสรหัสผ่านด้วย SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """ตรวจสอบรหัสผ่านว่าตรงกับ hash หรือไม่"""
    return hash_password(password) == hashed

def login_required(f):
    """decorator ตรวจสอบว่าล็อกอินแล้ว ถ้ายัง redirect ไปหน้า login"""
    @wraps(f) #คำสั่ง @waprs ทำหน้าที่ยืนยันฟังก์ชั่น
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('กรุณาล็อกอินก่อนเข้าสู่ระบบ', 'warning')
            return redirect(url_for('admin_login')) #url_for คือการสร้าง URL แบบไดนามิก
        return f(*args, **kwargs)
    return decorated_function

def create_admin_user():
    """สร้าง admin"""
    try:
        print("DEBUG: Creating admin user...")
        connection = get_db_connection()
        if not connection:
            print("DEBUG: Database connection failed in create_admin_user")
            return False
        
        print("DEBUG: Database connection successful in create_admin_user")
        cursor = connection.cursor()
        
        print("DEBUG: Checking admin user with PostgreSQL")
        cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        # เงื่อนไขเช็คว่ามี admin แล้วหรือยังหากยังไม่มี ให้สร้าง admin -ขึ้นมาใหม่ หากมีแล้วให้ print ว่ามีแล้ว
        if not admin_exists:
            # สร้าง admin
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            # เพิ่มข้อมูล admin ที่ยังไม่มีไปยัง ตาราง admin_users
            cursor.execute("""
                INSERT INTO admin_users (username, password_hash)
                VALUES ('admin', %s)
            """, (admin_password,))
            
            print("DEBUG: Admin user created: admin / admin123")
        else:
            print("DEBUG: Admin user already exists")
        
        connection.commit()
        cursor.close()
        connection.close()
        print("DEBUG: Admin user creation completed successfully")
        return True
        
    except psycopg2.Error as e:
        print(f"DEBUG: PostgreSQL error in create_admin_user: {e}")
        return False
    except Exception as e:
        print(f"DEBUG: General error in create_admin_user: {e}")
        return False

# ==================== Routes - หน้าหลัก / สมาชิก ====================
@app.route('/', methods=['GET'])
def index():
    """หน้าหลัก แสดงตารางคะแนนสมาชิกและสถิติรวม"""
    try:
        # สร้างตารางถ้าไม่มี
        init_database()
        
        connection = get_db_connection()

        # ส่งค่ากลับเป็น json 
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # ดึงข้อมูลสมาชิก (ไม่ใช้ JOIN เพื่อหลีกเลี่ยง error ตาราง member_names)
        cursor.execute("""
            SELECT rfid_id, SUM(score) as total_score,
                COUNT(CASE WHEN image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                MAX(scan_time) as last_scan
            FROM scan_logs
            GROUP BY rfid_id
            ORDER BY total_score DESC, scan_count DESC
        """)
        
        # ดึงข้อมูลทั้งหมดมาโชว์
        members = cursor.fetchall()
        
        # ดึงชื่อจาก member_names แยก (ถ้ามีตาราง)
        names_map = {}
        try:
            cursor.execute("SELECT rfid_id, display_name FROM member_names")
            for row in cursor.fetchall():
                # คำสั่ง d คือการแปลง row เป็น dict โดยตรวจสอบว่ามี method keys หรือไม่ 
                d = dict(row) if hasattr(row, 'keys') else {'rfid_id': row[0], 'display_name': row[1] if len(row) > 1 else None}
                if d.get('rfid_id'):
                    # แปลงเลขเป็น string
                    names_map[str(d['rfid_id'])] = d.get('display_name')
        except Exception:
            pass
        
        members = [dict(row) for row in members]
        
        # สร้าง display_label: "ชื่อ RFID" หรือแค่ rfid_id
        for m in members:
            dn = names_map.get(str(m.get('rfid_id', '')))
            m['display_name'] = dn
            m['display_label'] = f"{dn} {dn}" if dn else dn
        
        # ตรวจสอบข้อมูลที่ได้
        print(f"DEBUG: Found {len(members)} members")
        for i, member in enumerate(members[:3]):  # แสดงแค่ 3 คนแรก
            print(f"DEBUG: Member {i+1}: {member}")
        
        # ตรวจสอบ raw data
        print(f"DEBUG: Raw members data: {members}")
        
        # สถิติรวม
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_members FROM scan_logs")
        total_members_row = cursor.fetchone()
        total_members = total_members_row['total_members'] if total_members_row else 0
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        total_score_row = cursor.fetchone()
        total_score = total_score_row['total_score'] if total_score_row else 0
        # เช็คเงื่อนไขว่ามีการแสกนหรือไม่ ถ้ามีให้ส่งตัวเลขกลับถ้าไม่มีให้ส่ง 0 กลับไป
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        total_scans_row = cursor.fetchone()
        total_scans = total_scans_row['total_scans'] if total_scans_row else 0
        
        cursor.close()
        connection.close()

        # Render กลับไปยัง HTML
        return render_template('members.html', 
                             members=members, 
                             total_members=total_members,
                             total_score=total_score,
                             total_scans=total_scans)
        
    except Exception as e:
        print(f"Index page error: {e}")
        return render_template('members.html', 
                             members=[], 
                             total_members=0,
                             total_score=0,
                             total_scans=0)

# ==================== Routes - Admin (ล็อกอิน / จัดการ) ====================
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """หน้ารับล็อกอิน admin - ตรวจสอบ username/password กับ admin_users"""
    if request.method == 'GET':
        if session.get('admin_logged_in'):
            return redirect(url_for('admin'))
        return render_template('admin_login.html')
    
    
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        # laert กลับไปยังหน้า login ว่าข้อมูลไม่ครบถ้วน
        flash('กรุณากรอกชื่อผู้ใช้และรหัสผ่าน', 'error')
        return redirect(url_for('admin_login'))
    
    connection = None
    try:
        print(f"DEBUG: Attempting login for username: {username}")
        connection = get_db_connection()
        if not connection:
            print("DEBUG: Database connection failed")
            flash('ไม่สามารถเชื่อมต่อฐานข้อมูลได้', 'error')
            return redirect(url_for('admin_login'))
        
        print("DEBUG: Database connection successful")
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        print("DEBUG: Using PostgreSQL query")
        cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = %s", (username,))
        
        user = cursor.fetchone()
        print(f"DEBUG: User found: {user}")
        
        if user:
            user_dict = dict(user)
                
            print(f"DEBUG: Verifying password for user: {user_dict['username']}")
            password_valid = verify_password(password, user_dict['password_hash'])
            print(f"DEBUG: Password valid: {password_valid}")
            
            if password_valid:
                session['admin_logged_in'] = True
                session['admin_username'] = user_dict['username']
                session['admin_id'] = user_dict['id']
                print("DEBUG: Login successful")
                flash('เข้าสู่ระบบสำเร็จ', 'success')
                return redirect(url_for('admin'))
            else:
                print("DEBUG: Password verification failed")
                flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'error')
                return redirect(url_for('admin_login'))
        else:
            print("DEBUG: User not found")
            flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'error')
            return redirect(url_for('admin_login'))
            
    except psycopg2.Error as e:
        print(f"DEBUG: PostgreSQL error: {e}")
        flash('เกิดข้อผิดพลาดในการเข้าสู่ระบบ', 'error')
        return redirect(url_for('admin_login'))
    
    except Exception as e:
        print(f"DEBUG: General error: {e}")
        flash('เกิดข้อผิดพลาดในการเข้าสู่ระบบ', 'error')
        return redirect(url_for('admin_login'))
    
    finally:
        if connection:
            connection.close()

@app.route('/admin', methods=['GET'])
@login_required
def admin():
    """หน้าผู้ดูแลระบบ"""
    return render_template('admin.html')

@app.route('/admin/logout', methods=['GET'])
def admin_logout():
    """ออกจากระบบ"""
    session.clear()
    flash('ออกจากระบบเรียบร้อย', 'info')
    return redirect(url_for('admin_login'))

# ==================== API Debug ====================
@app.route('/api/debug/admin', methods=['GET'])
def debug_admin():
    """Debug สำหรับตรวจสอบ admin"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        # cursor_factory = RealDictCursor ทำให้ผลลัพธ์ที่ได้เป็น dict แทนที่จะเป็น tuple
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = 'admin'")
        
        admin_user = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as count FROM scan_logs")
        
        member_count = cursor.fetchone()
        
        connection.close()
        
        if admin_user:
            admin_user_dict = dict(admin_user)
        else:
            admin_user_dict = None
            
        if member_count:
            count_value = member_count['count']
        else:
            count_value = 0
        
        return jsonify({
            'admin_user': admin_user_dict,
            'member_count': count_value,
            'database_type': 'PostgreSQL'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/create-admin', methods=['POST'])
def create_admin_api():
    """API endpoint สำหรับสร้าง admin user"""
    try:
        print("DEBUG: Creating admin user via API...")
        connection = get_db_connection()

        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        if not admin_exists:
            # สร้าง admin
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            
            cursor.execute("""
                INSERT INTO admin_users (username, password_hash)
                VALUES ('admin', %s)
            """, (admin_password,))
            
            connection.commit()
            print("DEBUG: Admin user created successfully")
            cursor.close()
            connection.close()
            return jsonify({'success': True, 'message': 'Admin user created successfully'})
        
        else:
            print("DEBUG: Admin user already exists")
            cursor.close()
            connection.close()
            return jsonify({'success': True, 'message': 'Admin user already exists'})
        
    except Exception as e:
        print(f"DEBUG: Error creating admin user: {e}")
        return jsonify({'error': str(e)})

# ==================== API Debug Data ====================
@app.route('/api/debug_data', methods=['GET'])
def debug_data():
    """API สำหรับดูข้อมูลในฐานข้อมูล"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = cursor.fetchall()
        
        # ข้อมูลจาก scan_logs
        cursor.execute("""
            SELECT 
                rfid_id,
                SUM(score) as total_score,
                COUNT(CASE WHEN image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                MAX(scan_time) as last_scan,
                SUM(bottle_count) as total_bottles,
                SUM(can_count) as total_cans,
                SUM(cap_count) as total_caps,
                SUM(label_count) as total_labels
            FROM scan_logs 
            GROUP BY rfid_id 
            ORDER BY total_score DESC
        """)
        
        members_data = cursor.fetchall()
        
        members_data = [dict(row) for row in members_data]
        tables = [dict(table) for table in tables]
        
        # ประวัติการสแกนล่าสุด 10 ครั้ง
        cursor.execute("""
            SELECT 
                id,
                rfid_id,
                bottle_count,
                can_count,
                cap_count,
                label_count,
                score,
                scan_time
            FROM scan_logs 
            ORDER BY scan_time DESC 
            LIMIT 10
        """)
        
        recent_scans = cursor.fetchall()
        
        recent_scans = [dict(scan) for scan in recent_scans]
        
        # สถิติรวม
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_users FROM scan_logs")
        total_users_row = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        total_scans_row = cursor.fetchone()
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        total_score_row = cursor.fetchone()
        
        total_users = total_users_row['total_users'] if total_users_row else 0
        total_scans = total_scans_row['total_scans'] if total_scans_row else 0
        total_score = total_score_row['total_score'] if total_score_row else 0
        
        # ข้อมูล admin users
        cursor.execute("SELECT username, created_at FROM admin_users")
        admin_users = cursor.fetchall()
        
        admin_users = [dict(user) for user in admin_users]
        
        database_type = "PostgreSQL"
        
        connection.close()
        
        return jsonify({
            'success': True,
            'database_type': database_type,
            'tables': [table['table_name'] for table in tables],
            'statistics': {
                'total_users': total_users,
                'total_scans': total_scans,
                'total_score': total_score
            },
            'members_data': members_data,
            'recent_scans': recent_scans,
            'admin_users': admin_users,
            'message': 'Debug data retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve debug data'
        })

# ==================== Routes - สถิติ / จัดการสมาชิก / ประวัติ ====================
@app.route('/stats', methods=['GET'])
@login_required
def stats():
    """หน้าสถิติ: จำนวนสมาชิก การสแกนรวม คะแนนรวม ผู้ใช้ 30 วันล่าสุด"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # สถิติสมาชิก (scan_logs)
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_members FROM scan_logs")
        result = cursor.fetchone()
        total_members = result['total_members'] if result else 0
        
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        result = cursor.fetchone()
        total_scans = result['total_scans'] if result else 0
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        result = cursor.fetchone()
        total_score = result['total_score'] if result else 0
        
        # ผู้ใช้งานจริง 30 วันล่าสุด 
        cursor.execute("""
            SELECT COUNT(DISTINCT rfid_id) as active_users 
            FROM scan_logs 
            WHERE scan_time >= NOW() - INTERVAL '30 days'
                AND image_path != 'ADMIN_ADJUSTMENT'
        """)
        result = cursor.fetchone()
        active_users = result['active_users'] if result else 0
        
        stats_data = {
            'total_members': total_members or 0,
            'total_scans': total_scans or 0,
            'total_score': total_score or 0,
            'active_users': active_users or 0
        }
        
        connection.close()
        return render_template('admin_stats.html', stats=stats_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manage_members', methods=['GET'])
@login_required
def manage_members():
    """หน้าจัดการสมาชิก: แก้ไขคะแนน แก้ชื่อ ลบสมาชิก"""
    try:
        init_database()
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # ดึงข้อมูลสมาชิก (ไม่ใช้ JOIN)
        cursor.execute("""
            SELECT s.rfid_id, COALESCE(SUM(s.score), 0) as total_score,
                   COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                   MAX(s.scan_time) as last_scan
            FROM scan_logs s
            GROUP BY s.rfid_id
            ORDER BY total_score DESC, scan_count DESC
        """)
        
        members = cursor.fetchall()
        
        # ดึงชื่อจาก member_names แยก
        names_map = {}
        try:
            cursor.execute("SELECT rfid_id, display_name FROM member_names")
            for row in cursor.fetchall():
                d = dict(row) if hasattr(row, 'keys') else {'rfid_id': row[0], 'display_name': row[1] if len(row) > 1 else None}
                if d.get('rfid_id'):
                    names_map[str(d['rfid_id'])] = d.get('display_name')
        except Exception:
            pass
        
        members = [dict(member) for member in members]
        
        for m in members:
            dn = names_map.get(str(m.get('rfid_id', '')))

            rfid_str = str(m.get('rfid_id', ''))

            if rfid_str == "718597286205":
                dn = 'Opal'

            elif rfid_str == "1070128781870":
                dn = 'Peem'

            m['display_name'] = dn
            m['display_label'] = f"{dn} {m['rfid_id']}" if dn else m['rfid_id']
        
        connection.close()
        return render_template('admin_members.html', members=members)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scan_history', methods=['GET'])
@login_required
def scan_history():
    """หน้าประวัติการสแกน 100 รายการล่าสุด (ไม่รวม ADMIN_ADJUSTMENT)"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT s.*, s.rfid_id as full_name, s.rfid_id as username
            FROM scan_logs s
            WHERE s.image_path != 'ADMIN_ADJUSTMENT'
            ORDER BY s.scan_time DESC
            LIMIT 100
        """)
        
        scan_history = cursor.fetchall()
        
        scan_history = [dict(scan) for scan in scan_history]
        
        # จัดรูปเป็น string
        for scan in scan_history:
            if scan.get('scan_time'):
                if isinstance(scan['scan_time'], str):
                    # ถ้าเป็น string แล้ว ให้แปลงเป็น datetime
                    try:
                        # ใช้ ไลบรารี่ datetime ในการแปลง string เป็น datetime โดยรองรับรูปแบบทั่วไป
                        from datetime import datetime
                        # ถ้าเจอ ตัว T ใหญ่ ใน scan_time แสดงว่าเป็นรูปแบบ ให้แปลงแบบ
                        if 'T' in scan['scan_time']:
                            dt = datetime.fromisoformat(scan['scan_time'].replace('Z', '+00:00'))

                        else:
                            dt = datetime.strptime(scan['scan_time'], '%Y-%m-%d %H:%M:%S')
                        scan['scan_time'] = dt.strftime('%d/%m/%Y %H:%M')
                        
                    except:
                        scan['scan_time'] = str(scan['scan_time'])

                else:
                    # ถ้าเป็น datetime object
                    scan['scan_time'] = scan['scan_time'].strftime('%d/%m/%Y %H:%M')
        
        connection.close()
        return render_template('admin_history.html', scan_history=scan_history)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== API Admin (แก้ไขคะแนน / ชื่อ / ลบสมาชิก) ====================
@app.route('/api/admin/edit-score', methods=['POST'])
@login_required
def admin_edit_score():
    """แก้ไขคะแนนรวมสมาชิก โดยเพิ่ม record ADMIN_ADJUSTMENT เพื่อให้คะแนนรวมตรงกับค่าที่ตั้ง"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        new_score = data.get('new_score')
        
        if not rfid_id or new_score is None:
            return jsonify({'success': False, 'message': 'ข้อมูลไม่ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            "SELECT SUM(score) as current_total FROM scan_logs WHERE rfid_id = %s",
            (rfid_id,),
        )
        result = cursor.fetchone()
        current_total = result['current_total'] if result and result['current_total'] else 0
        score_difference = new_score - current_total
        
        cursor.execute("""
            INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (rfid_id, 0, 0, 0, 0, score_difference, 'ADMIN_ADJUSTMENT'))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'แก้ไขคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})



@app.route('/api/admin/edit-member-name', methods=['POST'])
@login_required
def admin_edit_member_name():
    """เพิ่ม/แก้ไขชื่อแสดงใน member_names (ใช้เป็น display_label แบบ "ชื่อ RFID")"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        display_name = (data.get('display_name') or '').strip() # .strip() ตัดช่องว่างทิ้งออก ของ String
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()

        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()

        target_rfid = str(rfid_id)

        if target_rfid == "718597286205":
            display_name = 'Opal'
        elif target_rfid == "1070128781870":
            display_name = 'Peem'

        try:
            if display_name:
                cursor.execute("""
                    INSERT INTO member_names (rfid_id, display_name)
                    VALUES (%s, %s)
                    ON CONFLICT (rfid_id) DO UPDATE SET display_name = EXCLUDED.display_name
                """, (display_name , display_name))

            else:
                cursor.execute("DELETE FROM member_names WHERE rfid_id = %s", (rfid_id,))

        except Exception as e:
            connection.rollback()
            return jsonify({'success': False, 'message': f'ตาราง member_names ยังไม่พร้อม: {str(e)}'})
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'แก้ไขชื่อสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})


@app.route('/api/admin/delete-member', methods=['POST'])
@login_required
def admin_delete_member():
    """ลบข้อมูลสมาชิกจาก scan_logs และ member_names"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')

        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()

        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        cursor.execute("DELETE FROM scan_logs WHERE rfid_id = %s", (rfid_id,))
        try:
            # ลบรายชื้ออกจากตาราง member_names
            cursor.execute("DELETE FROM member_names WHERE rfid_id = %s", (rfid_id,))
        except Exception:
            pass
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'ลบสมาชิกสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

def sum_score(bottle_count, can_count, cap_count ,label_count):
    score = (bottle_count * 50) + (can_count * 100) - (cap_count * 10) - (label_count * 10)
    return score

# ==================== API Pi Client (รับคะแนนจากการสแกน) ====================
@app.route('/api/add_score', methods=['POST'])
def add_score():
    """รับข้อมูลจาก Pi หลังสแกน บันทึกลง scan_logs คำนวณ score จาก bottle/can/cap/label"""
    try:
        # สร้างตารางถ้ายังไม่มี
        init_database()
        
        # รับข้อมูล JSON
        if request.is_json:
            data = request.get_json() or {}
        else:
            try:
                data = request.get_json(force=True) or {}
            except:
                data = {}
        
        rfid_id = data.get('card_id') or data.get('rfid_id')
        bottle_count = data.get('bottle_count', 0)
        can_count = data.get('can_count', 0)
        cap_count = data.get('cap_count', 0)
        label_count = data.get('label_count', 0)
        image_path = data.get('image_path', '')
        
        # คำนวณคะแนน: ขวด+50 กระป๋อง+100 แก๊ป-10 ฉลาก-10
        score = sum_score(bottle_count, can_count, cap_count ,label_count)
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()

        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        # ใส่ข้อมูลไปยังฐานข้อมูล โดยแยกคอลัมน์สำหรับ bottle_count, can_count, cap_count, label_count เพื่อให้สามารถคำนวณคะแนนได้ง่ายขึ้นในภายหลัง
        cursor.execute("""
            INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'บันทึกคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})


# ==================== Error Handler ====================
@app.errorhandler(500)
def internal_error(e):
    """แสดง error จริงเมื่อเกิด 500 (เพื่อ debug)"""
    import traceback
    tb = traceback.format_exc()
    return f'<pre style="white-space:pre-wrap;font-size:12px;">Internal Server Error\n\n{tb}</pre>', 500


# ==================== Main ====================
if __name__ == '__main__':
    if init_database():
        print("Database พร้อม")
        
        create_admin_user()
        # Get port
        port = int(os.environ.get('PORT', 9000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Database initialization failed")
        print("Please check PostgreSQL server and configuration")