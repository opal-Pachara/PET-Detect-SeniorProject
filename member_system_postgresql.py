#!/usr/bin/env python3
"""
PET Detect Member System - PostgreSQL Version for Render
รองรับ Render, Fly.io และ Cloud Platforms อื่นๆ
"""

import os
import hashlib
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pet_detect_secret_key_2025')

# Database configuration for cloud deployment
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'pet_detect_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'password')
}

# Debug: Print database configuration (without password)
print(f"Database Config: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']} (user: {DB_CONFIG['user']})")

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล PostgreSQL หรือ SQLite fallback"""
    try:
        # ลองเชื่อมต่อ PostgreSQL ก่อน
        connection = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL")
        return connection
    except psycopg2.Error as e:
        print(f"PostgreSQL connection failed: {e}")
        try:
            # Fallback to SQLite
            connection = sqlite3.connect('pet_detect.db')
            # Set row factory สำหรับ SQLite
            connection.row_factory = sqlite3.Row
            print("Connected to SQLite (fallback)")
            return connection
        except Exception as sqlite_error:
            print(f"SQLite connection also failed: {sqlite_error}")
            return None

def init_database():
    """สร้างตารางฐานข้อมูล"""
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # ตรวจสอบว่าเป็น PostgreSQL หรือ SQLite
        if isinstance(connection, psycopg2.extensions.connection):
            # PostgreSQL syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    id SERIAL PRIMARY KEY,
                    rfid_id VARCHAR(50) UNIQUE NOT NULL,
                    username VARCHAR(100),
                    password_hash VARCHAR(255),
                    full_name VARCHAR(200),
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    total_score INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
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
        else:
            # SQLite syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rfid_id TEXT UNIQUE NOT NULL,
                    username TEXT,
                    password_hash TEXT,
                    full_name TEXT,
                    email TEXT,
                    phone TEXT,
                    total_score INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rfid_id TEXT NOT NULL,
                    bottle_count INTEGER DEFAULT 0,
                    can_count INTEGER DEFAULT 0,
                    cap_count INTEGER DEFAULT 0,
                    label_count INTEGER DEFAULT 0,
                    score INTEGER DEFAULT 0,
                    image_path TEXT,
                    scan_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("Database tables created successfully!")
        return True
        
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False

def hash_password(password):
    """เข้ารหัสรหัสผ่าน"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """ตรวจสอบรหัสผ่าน"""
    return hash_password(password) == hashed

def login_required(f):
    """Decorator สำหรับตรวจสอบการล็อกอิน"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('กรุณาล็อกอินก่อนเข้าสู่ระบบ', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def create_admin_user():
    """สร้างผู้ดูแลระบบเริ่มต้น"""
    try:
        print("DEBUG: Creating admin user...")
        connection = get_db_connection()
        if not connection:
            print("DEBUG: Database connection failed in create_admin_user")
            return False
        
        print("DEBUG: Database connection successful in create_admin_user")
        cursor = connection.cursor()
        
        # ตรวจสอบว่ามี admin หรือไม่
        if isinstance(connection, psycopg2.extensions.connection):
            print("DEBUG: Checking admin user with PostgreSQL")
            cursor.execute("SELECT id FROM members WHERE username = 'admin'")
        else:
            print("DEBUG: Checking admin user with SQLite")
            cursor.execute("SELECT id FROM members WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        if not admin_exists:
            # สร้าง admin user
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                    VALUES ('ADMIN001', 'admin', %s, 'System Administrator', 'admin@petdetect.com', '000-000-0000')
                """, (admin_password,))
            else:
                cursor.execute("""
                    INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                    VALUES ('ADMIN001', 'admin', ?, 'System Administrator', 'admin@petdetect.com', '000-000-0000')
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

@app.route('/', methods=['GET'])
def index():
    """หน้าหลัก - แสดงตารางคะแนนเลย"""
    try:
        # สร้างตารางถ้ายังไม่มี
        init_database()
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # ใช้ cursor ที่เหมาะสมกับ database
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            # SQLite (row_factory already set in get_db_connection)
            cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิกทั้งหมดเรียงตามคะแนน
        if isinstance(connection, psycopg2.extensions.connection):
            # PostgreSQL syntax - แสดงข้อมูลจาก scan_logs แม้ไม่มี members
            cursor.execute("""
                SELECT 
                    s.rfid_id,
                    COALESCE(m.full_name, s.rfid_id) as full_name,
                    COALESCE(m.username, s.rfid_id) as username,
                    COALESCE(m.email, '') as email,
                    SUM(s.score) as total_score,
                    COUNT(s.id) as scan_count,
                    MAX(s.scan_time) as last_scan
                FROM scan_logs s
                LEFT JOIN members m ON s.rfid_id = m.rfid_id
                GROUP BY s.rfid_id
                ORDER BY total_score DESC, scan_count DESC
            """)
        else:
            # SQLite syntax - แสดงข้อมูลจาก scan_logs แม้ไม่มี members
            cursor.execute("""
                SELECT 
                    s.rfid_id,
                    COALESCE(m.full_name, s.rfid_id) as full_name,
                    COALESCE(m.username, s.rfid_id) as username,
                    COALESCE(m.email, '') as email,
                    SUM(s.score) as total_score,
                    COUNT(s.id) as scan_count,
                    MAX(s.scan_time) as last_scan
                FROM scan_logs s
                LEFT JOIN members m ON s.rfid_id = m.rfid_id
                GROUP BY s.rfid_id
                ORDER BY total_score DESC, scan_count DESC
            """)
        
        members = cursor.fetchall()
        
        # แปลง SQLite Row objects เป็น dict
        if isinstance(connection, sqlite3.Connection):
            members = [dict(row) for row in members]
        
        # Debug: ตรวจสอบข้อมูลที่ได้
        print(f"DEBUG: Found {len(members)} members")
        for i, member in enumerate(members[:3]):  # แสดงแค่ 3 คนแรก
            print(f"DEBUG: Member {i+1}: {member}")
        
        # Debug: ตรวจสอบ raw data
        print(f"DEBUG: Raw members data: {members}")
        
        # สถิติรวม (รองรับทั้ง PostgreSQL และ SQLite)
        cursor.execute("SELECT COUNT(*) as total_members FROM members")
        total_members_row = cursor.fetchone()
        if isinstance(connection, sqlite3.Connection):
            total_members = total_members_row['total_members'] if total_members_row else 0
        else:
            total_members = total_members_row['total_members'] if total_members_row else 0
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        total_score_row = cursor.fetchone()
        if isinstance(connection, sqlite3.Connection):
            total_score = total_score_row['total_score'] if total_score_row else 0
        else:
            total_score = total_score_row['total_score'] if total_score_row else 0
        
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        total_scans_row = cursor.fetchone()
        if isinstance(connection, sqlite3.Connection):
            total_scans = total_scans_row['total_scans'] if total_scans_row else 0
        else:
            total_scans = total_scans_row['total_scans'] if total_scans_row else 0
        
        cursor.close()
        connection.close()
        
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

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """หน้าล็อกอินผู้ดูแลระบบ"""
    if request.method == 'GET':
        if session.get('admin_logged_in'):
            return redirect(url_for('admin'))
        return render_template('admin_login.html')
    
    # POST method - จัดการ form submission
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        flash('กรุณากรอกชื่อผู้ใช้และรหัสผ่าน', 'error')
        return redirect(url_for('admin_login'))
    
    try:
        print(f"DEBUG: Attempting login for username: {username}")
        connection = get_db_connection()
        if not connection:
            print("DEBUG: Database connection failed")
            flash('ไม่สามารถเชื่อมต่อฐานข้อมูลได้', 'error')
            return redirect(url_for('admin_login'))
        
        print("DEBUG: Database connection successful")
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ตรวจสอบผู้ใช้
        if isinstance(connection, psycopg2.extensions.connection):
            print("DEBUG: Using PostgreSQL query")
            cursor.execute("SELECT id, username, password_hash FROM members WHERE username = %s", (username,))
        else:
            print("DEBUG: Using SQLite query")
            cursor.execute("SELECT id, username, password_hash FROM members WHERE username = ?", (username,))
        
        user = cursor.fetchone()
        print(f"DEBUG: User found: {user}")
        
        if user:
            # แปลง user เป็น dict สำหรับ SQLite
            if isinstance(connection, sqlite3.Connection):
                user_dict = dict(user)
            else:
                user_dict = user
                
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
    """หน้าผู้ดูแลระบบ - ต้องล็อกอิน"""
    return render_template('admin.html')

@app.route('/admin/logout', methods=['GET'])
def admin_logout():
    """ออกจากระบบ"""
    session.clear()
    flash('ออกจากระบบเรียบร้อย', 'info')
    return redirect(url_for('index'))

@app.route('/api/debug/admin', methods=['GET'])
def debug_admin():
    """Debug endpoint สำหรับตรวจสอบ admin user"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ตรวจสอบ admin user
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT id, username, password_hash, full_name FROM members WHERE username = 'admin'")
        else:
            cursor.execute("SELECT id, username, password_hash, full_name FROM members WHERE username = 'admin'")
        
        admin_user = cursor.fetchone()
        
        # ตรวจสอบจำนวนสมาชิกทั้งหมด
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT COUNT(*) as count FROM members")
        else:
            cursor.execute("SELECT COUNT(*) as count FROM members")
        
        member_count = cursor.fetchone()
        
        # ตรวจสอบ database type ก่อนปิด connection
        is_postgresql = isinstance(connection, psycopg2.extensions.connection)
        
        connection.close()
        
        # แปลงข้อมูลให้เหมาะสมกับ database type
        if admin_user:
            if is_postgresql:
                admin_user_dict = dict(admin_user)
            else:
                admin_user_dict = dict(admin_user)
        else:
            admin_user_dict = None
            
        if member_count:
            if is_postgresql:
                count_value = member_count['count']
            else:
                count_value = member_count['count']
        else:
            count_value = 0
        
        return jsonify({
            'admin_user': admin_user_dict,
            'member_count': count_value,
            'database_type': 'PostgreSQL' if is_postgresql else 'SQLite'
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
        
        # ตรวจสอบว่ามี admin หรือไม่
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT id FROM members WHERE username = 'admin'")
        else:
            cursor.execute("SELECT id FROM members WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        if not admin_exists:
            # สร้าง admin user
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                    VALUES ('ADMIN001', 'admin', %s, 'System Administrator', 'admin@petdetect.com', '000-000-0000')
                """, (admin_password,))
            else:
                cursor.execute("""
                    INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                    VALUES ('ADMIN001', 'admin', ?, 'System Administrator', 'admin@petdetect.com', '000-000-0000')
                """, (admin_password,))
            
            connection.commit()
            print("DEBUG: Admin user created successfully")
            return jsonify({'success': True, 'message': 'Admin user created successfully'})
        else:
            print("DEBUG: Admin user already exists")
            return jsonify({'success': True, 'message': 'Admin user already exists'})
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"DEBUG: Error creating admin user: {e}")
        return jsonify({'error': str(e)})


@app.route('/stats', methods=['GET'])
@login_required
def stats():
    """หน้าสถิติการใช้งานระบบ"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # สถิติสมาชิก
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT COUNT(*) as total_members FROM members")
            total_members = cursor.fetchone()['total_members']
            
            cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
            total_scans = cursor.fetchone()['total_scans']
            
            cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
            total_score = cursor.fetchone()['total_score']
            
            cursor.execute("SELECT COUNT(DISTINCT rfid_id) as active_users FROM scan_logs")
            active_users = cursor.fetchone()['active_users']
        else:
            cursor.execute("SELECT COUNT(*) as total_members FROM members")
            total_members = cursor.fetchone()['total_members']
            
            cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
            total_scans = cursor.fetchone()['total_scans']
            
            cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
            total_score = cursor.fetchone()['total_score']
            
            cursor.execute("SELECT COUNT(DISTINCT rfid_id) as active_users FROM scan_logs")
            active_users = cursor.fetchone()['active_users']
        
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
    """หน้าจัดการสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิกจาก scan_logs (แสดงทุกคนที่มีการสแกน)
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT s.rfid_id,
                       COALESCE(SUM(s.score), 0) as total_score,
                       COUNT(s.id) as scan_count,
                       MAX(s.scan_time) as last_scan
                FROM scan_logs s
                GROUP BY s.rfid_id
                ORDER BY total_score DESC, scan_count DESC
            """)
        else:
            cursor.execute("""
                SELECT s.rfid_id,
                       COALESCE(SUM(s.score), 0) as total_score,
                       COUNT(s.id) as scan_count,
                       MAX(s.scan_time) as last_scan
                FROM scan_logs s
                GROUP BY s.rfid_id
                ORDER BY total_score DESC, scan_count DESC
            """)
        
        members = cursor.fetchall()
        
        # แปลงข้อมูลสำหรับ SQLite
        if isinstance(connection, sqlite3.Connection):
            members = [dict(member) for member in members]
        
        connection.close()
        return render_template('admin_members.html', members=members)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scan_history', methods=['GET'])
@login_required
def scan_history():
    """หน้าประวัติการสแกน"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ดึงประวัติการสแกนทั้งหมด
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT s.*, m.full_name, m.username
                FROM scan_logs s
                LEFT JOIN members m ON s.rfid_id = m.rfid_id
                ORDER BY s.scan_time DESC
                LIMIT 100
            """)
        else:
            cursor.execute("""
                SELECT s.*, m.full_name, m.username
                FROM scan_logs s
                LEFT JOIN members m ON s.rfid_id = m.rfid_id
                ORDER BY s.scan_time DESC
                LIMIT 100
            """)
        
        scan_history = cursor.fetchall()
        
        # แปลงข้อมูลสำหรับ SQLite และจัดรูปแบบวันที่
        if isinstance(connection, sqlite3.Connection):
            scan_history = [dict(scan) for scan in scan_history]
        
        # จัดรูปแบบวันที่ให้เป็น string
        for scan in scan_history:
            if scan.get('scan_time'):
                if isinstance(scan['scan_time'], str):
                    # ถ้าเป็น string แล้ว ให้แปลงเป็น datetime object ก่อน
                    try:
                        from datetime import datetime
                        if 'T' in scan['scan_time']:
                            # ISO format
                            dt = datetime.fromisoformat(scan['scan_time'].replace('Z', '+00:00'))
                        else:
                            # SQLite format
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


@app.route('/api/admin/edit-score', methods=['POST'])
@login_required
def admin_edit_score():
    """แก้ไขคะแนนสมาชิก"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        new_score = data.get('new_score')
        
        if not rfid_id or new_score is None:
            return jsonify({'success': False, 'message': 'ข้อมูลไม่ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # อัปเดตคะแนนใน scan_logs (เพิ่มคะแนนให้กับรายการล่าสุด)
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                UPDATE scan_logs 
                SET score = %s 
                WHERE rfid_id = %s 
                AND scan_time = (SELECT MAX(scan_time) FROM scan_logs WHERE rfid_id = %s)
            """, (new_score, rfid_id, rfid_id))
        else:
            cursor.execute("""
                UPDATE scan_logs 
                SET score = ? 
                WHERE rfid_id = ? 
                AND scan_time = (SELECT MAX(scan_time) FROM scan_logs WHERE rfid_id = ?)
            """, (new_score, rfid_id, rfid_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'แก้ไขคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/admin/delete-member', methods=['POST'])
@login_required
def admin_delete_member():
    """ลบสมาชิก"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # ลบข้อมูลจาก scan_logs (ไม่ต้องลบจาก members เพราะแสดงจาก scan_logs)
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("DELETE FROM scan_logs WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("DELETE FROM scan_logs WHERE rfid_id = ?", (rfid_id,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'ลบสมาชิกสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})





@app.route('/api/add_score', methods=['POST'])
def add_score():
    """เพิ่มคะแนนจาก RFID scan"""
    try:
        # สร้างตารางถ้ายังไม่มี
        init_database()
        
        data = request.json
        rfid_id = data.get('card_id') or data.get('rfid_id')
        bottle_count = data.get('bottle_count', 0)
        can_count = data.get('can_count', 0)
        cap_count = data.get('cap_count', 0)
        label_count = data.get('label_count', 0)
        image_path = data.get('image_path', '')
        
        # คำนวณคะแนนใหม่ตามระบบใหม่
        # ขวด PET: +50 คะแนน, กระป๋อง: +100 คะแนน, ฝา: -10 คะแนน, สลาก: -10 คะแนน
        score = (bottle_count * 50) + (can_count * 100) - (cap_count * 10) - (label_count * 10)
        
        # Debug logging
        print(f"DEBUG: Received data: {data}")
        print(f"DEBUG: RFID ID: {rfid_id}")
        print(f"DEBUG: Score: {score}")
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # เพิ่มข้อมูลการสแกน (รองรับทั้ง PostgreSQL และ SQLite)
        if isinstance(connection, sqlite3.Connection):
            # SQLite syntax
            cursor.execute("""
                INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        else:
            # PostgreSQL syntax
            cursor.execute("""
                INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        
        # อัปเดตคะแนนรวมในตาราง members
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                UPDATE members 
                SET total_score = (
                    SELECT COALESCE(SUM(score), 0) 
                    FROM scan_logs 
                    WHERE rfid_id = %s
                )
                WHERE rfid_id = %s
            """, (rfid_id, rfid_id))
        else:
            cursor.execute("""
                UPDATE members 
                SET total_score = (
                    SELECT COALESCE(SUM(score), 0) 
                    FROM scan_logs 
                    WHERE rfid_id = ?
                )
                WHERE rfid_id = ?
            """, (rfid_id, rfid_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"DEBUG: Successfully saved score for RFID {rfid_id}")
        return jsonify({'success': True, 'message': 'บันทึกคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})


if __name__ == '__main__':
    print("PET Detect Member System - PostgreSQL Version")
    print("=" * 50)
    print("Available Routes:")
    print("   - GET  /                    - หน้าแสดงตารางคะแนน")
    print("   - GET/POST /admin/login     - ล็อกอินผู้ดูแลระบบ")
    print("   - GET  /admin               - หน้าผู้ดูแลระบบ")
    print("   - GET  /admin/logout        - ออกจากระบบ")
    print("   - GET  /stats               - หน้าสถิติระบบ")
    print("   - GET  /manage_members      - จัดการสมาชิก")
    print("   - GET  /scan_history        - ประวัติการสแกน")
    print("   - POST /api/add_score       - เพิ่มคะแนนจาก Pi")
    print("   - POST /api/admin/edit-score - แก้ไขคะแนน")
    print("   - POST /api/admin/delete-member - ลบสมาชิก")
    print("   - GET  /api/debug/admin     - ตรวจสอบ admin user")
    print("   - POST /api/create-admin    - สร้าง admin user")
    print("Press Ctrl+C to stop")
    
    # Initialize database
    if init_database():
        print("Database ready!")
        # Create admin user
        create_admin_user()
        # Get port from environment variable (for cloud deployment)
        port = int(os.environ.get('PORT', 9000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Database initialization failed!")
        print("Please check PostgreSQL server and configuration")
