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
    """หน้าหลัก - แสดงอันดับสมาชิก"""
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

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """หน้า Dashboard"""
    return render_template('members.html')

@app.route('/members', methods=['GET'])
def members():
    """หน้าอันดับ - redirect ไปหน้าหลัก"""
    return redirect(url_for('index'))

@app.route('/register', methods=['GET'])
def register_page():
    """หน้าสมัครสมาชิก - เริ่มต้นด้วย RFID"""
    return render_template('register.html')

@app.route('/register/rfid', methods=['GET'])
def register_rfid():
    """หน้าล็อกด้วย RFID"""
    return render_template('register_rfid.html')

@app.route('/register/check', methods=['POST'])
def register_check():
    """ตรวจสอบ RFID และนำทางไปหน้าถัดไป"""
    try:
        rfid_id = request.form.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'กรุณากรอก RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # ตรวจสอบว่า RFID ID มีข้อมูลการสแกนหรือไม่
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE rfid_id = ?", (rfid_id,))
        
        scan_count = cursor.fetchone()[0]
        
        if scan_count == 0:
            # ไม่มีข้อมูลการสแกน -> ไม่ให้สมัคร
            cursor.close()
            connection.close()
            return jsonify({
                'success': False, 
                'message': 'ไม่พบข้อมูลการสแกน กรุณาแตะบัตร RFID ก่อน'
            })
        
        # ตรวจสอบว่า RFID ID มีสมาชิกอยู่แล้วหรือไม่
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT id, username, password_hash, full_name, email FROM members WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("SELECT id, username, password_hash, full_name, email FROM members WHERE rfid_id = ?", (rfid_id,))
        
        member = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if member:
            # มีสมาชิกอยู่แล้ว
            has_password = bool(member[2]) if member[2] else False
            
            if has_password:
                # มีรหัสผ่านแล้ว -> ไปหน้าล็อกอิน
                return jsonify({
                    'success': True,
                    'action': 'login',
                    'rfid_id': rfid_id,
                    'message': 'พบสมาชิก กรุณาล็อกอิน'
                })
            else:
                # ยังไม่มีรหัสผ่าน -> ไปหน้าสร้างรหัสผ่าน
                return jsonify({
                    'success': True,
                    'action': 'create_password',
                    'rfid_id': rfid_id,
                    'username': member[1],
                    'message': 'กรุณาสร้างรหัสผ่าน'
                })
        else:
            # ยังไม่ใช่สมาชิก แต่มีข้อมูลการสแกน -> ไปหน้าสมัครสมาชิกใหม่
            return jsonify({
                'success': True,
                'action': 'register_new',
                'rfid_id': rfid_id,
                'message': 'พบข้อมูลการสแกน กรุณาสมัครสมาชิกใหม่'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/register/new', methods=['GET', 'POST'])
def register_new_member():
    """สมัครสมาชิกใหม่"""
    if request.method == 'GET':
        # แสดงหน้า form
        rfid_id = request.args.get('rfid_id')
        return render_template('register_new.html', rfid_id=rfid_id)
    
    # POST method - จัดการ form submission
    try:
        rfid_id = request.form.get('rfid_id')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        
        if not all([rfid_id, password, full_name, email]):
            return jsonify({'success': False, 'message': 'กรุณากรอกข้อมูลให้ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # สร้าง username จาก RFID
        username = f"user_{rfid_id[:8]}"
        password_hash = hash_password(password)
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                INSERT INTO members (rfid_id, username, password_hash, full_name, email)
                VALUES (%s, %s, %s, %s, %s)
            """, (rfid_id, username, password_hash, full_name, email))
        else:
            cursor.execute("""
                INSERT INTO members (rfid_id, username, password_hash, full_name, email)
                VALUES (?, ?, ?, ?, ?)
            """, (rfid_id, username, password_hash, full_name, email))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'สมัครสมาชิกสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/register/password', methods=['GET', 'POST'])
def register_create_password():
    """สร้างรหัสผ่านสำหรับสมาชิกที่มีอยู่"""
    if request.method == 'GET':
        # แสดงหน้า form
        rfid_id = request.args.get('rfid_id')
        username = request.args.get('username')
        return render_template('register_password.html', rfid_id=rfid_id, username=username)
    
    # POST method - จัดการ form submission
    try:
        rfid_id = request.form.get('rfid_id')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        
        if not all([rfid_id, password, full_name, email]):
            return jsonify({'success': False, 'message': 'กรุณากรอกข้อมูลให้ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        password_hash = hash_password(password)
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                UPDATE members 
                SET password_hash = %s, full_name = %s, email = %s, updated_at = CURRENT_TIMESTAMP
                WHERE rfid_id = %s
            """, (password_hash, full_name, email, rfid_id))
        else:
            cursor.execute("""
                UPDATE members 
                SET password_hash = ?, full_name = ?, email = ?, updated_at = CURRENT_TIMESTAMP
                WHERE rfid_id = ?
            """, (password_hash, full_name, email, rfid_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'สร้างรหัสผ่านสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/register/login', methods=['GET', 'POST'])
def register_login():
    """ล็อกอินสมาชิก"""
    if request.method == 'GET':
        # แสดงหน้า form
        rfid_id = request.args.get('rfid_id')
        return render_template('register_login.html', rfid_id=rfid_id)
    
    # POST method - จัดการ form submission
    try:
        rfid_id = request.form.get('rfid_id')
        password = request.form.get('password')
        
        if not all([rfid_id, password]):
            return jsonify({'success': False, 'message': 'กรุณากรอกข้อมูลให้ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        
        member = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if member and verify_password(password, member[2]):
            # ล็อกอินสำเร็จ
            return jsonify({
                'success': True,
                'message': 'เข้าสู่ระบบสำเร็จ',
                'member': {
                    'rfid_id': member[1],
                    'username': member[2],
                    'full_name': member[4],
                    'email': member[5]
                }
            })
        else:
            return jsonify({'success': False, 'message': 'รหัสผ่านไม่ถูกต้อง'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/member/<rfid_id>', methods=['GET'])
def member_detail(rfid_id):
    """รายละเอียดสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิก
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        
        member = cursor.fetchone()
        
        # ดึงประวัติการสแกน
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT * FROM scan_logs 
                WHERE rfid_id = %s 
                ORDER BY scan_time DESC
            """, (rfid_id,))
        else:
            cursor.execute("""
                SELECT * FROM scan_logs 
                WHERE rfid_id = ? 
                ORDER BY scan_time DESC
            """, (rfid_id,))
        
        scan_history = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # แปลง SQLite Row objects เป็น dict
        if isinstance(connection, sqlite3.Connection):
            if member:
                member = dict(member)
            scan_history = [dict(row) for row in scan_history]
        
        return render_template('member_detail.html', 
                             rfid_id=rfid_id, 
                             member=member, 
                             scan_history=scan_history)
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/member/<rfid_id>', methods=['GET'])
def api_get_member(rfid_id):
    """API ดึงข้อมูลสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิก
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        else:
            cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        
        member = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if member:
            # แปลง SQLite Row objects เป็น dict
            if isinstance(connection, sqlite3.Connection):
                member = dict(member)
            return jsonify({'success': True, 'member': member})
        else:
            return jsonify({'success': False, 'message': 'ไม่พบสมาชิก'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/member/<rfid_id>/history', methods=['GET'])
def api_get_member_history(rfid_id):
    """API ดึงประวัติการสแกนของสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # ดึงประวัติการสแกน
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT * FROM scan_logs 
                WHERE rfid_id = %s 
                ORDER BY scan_time DESC
            """, (rfid_id,))
        else:
            cursor.execute("""
                SELECT * FROM scan_logs 
                WHERE rfid_id = ? 
                ORDER BY scan_time DESC
            """, (rfid_id,))
        
        history = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # แปลง SQLite Row objects เป็น dict
        if isinstance(connection, sqlite3.Connection):
            history = [dict(row) for row in history]
        
        return jsonify({'success': True, 'history': history})
        
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
        score = data.get('score', 0)
        image_path = data.get('image_path', '')
        
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
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"DEBUG: Successfully saved score for RFID {rfid_id}")
        return jsonify({'success': True, 'message': 'บันทึกคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/check_member', methods=['GET'])
def check_member():
    """ตรวจสอบว่าสมาชิกมีอยู่หรือไม่"""
    try:
        rfid_id = request.args.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if member:
            # แปลง member เป็น dict สำหรับ SQLite
            if isinstance(connection, sqlite3.Connection):
                member_dict = dict(member)
            else:
                member_dict = member
            has_password = bool(member_dict['password_hash'])
            return jsonify({
                'success': True,
                'is_member': True,
                'has_password': has_password,
                'member': member_dict
            })
        else:
            return jsonify({
                'success': True,
                'is_member': False,
                'has_password': False
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/debug_data', methods=['GET'])
def debug_data():
    """ดูข้อมูลทั้งหมดในฐานข้อมูล (สำหรับ debug)"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        cursor = connection.cursor()
        
        # ดูข้อมูลใน members table
        cursor.execute("SELECT * FROM members")
        members = cursor.fetchall()
        
        # ดูข้อมูลใน scan_logs table
        cursor.execute("SELECT * FROM scan_logs ORDER BY scan_time DESC LIMIT 10")
        scan_logs = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        # แปลง SQLite Row objects เป็น dict
        if isinstance(connection, sqlite3.Connection):
            members = [dict(row) for row in members]
            scan_logs = [dict(row) for row in scan_logs]
        
        return jsonify({
            'success': True,
            'members': members,
            'scan_logs': scan_logs,
            'members_count': len(members),
            'scan_logs_count': len(scan_logs)
        })
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/verify_password', methods=['POST'])
def verify_member_password():
    """ตรวจสอบรหัสผ่านสมาชิก"""
    try:
        data = request.json
        rfid_id = data.get('rfid_id')
        password = data.get('password')
        
        if not all([rfid_id, password]):
            return jsonify({'success': False, 'message': 'กรุณากรอกข้อมูลให้ครบถ้วน'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if member:
            # แปลง member เป็น dict สำหรับ SQLite
            if isinstance(connection, sqlite3.Connection):
                member_dict = dict(member)
            else:
                member_dict = member
                
            if verify_password(password, member_dict['password_hash']):
                return jsonify({'success': True, 'message': 'รหัสผ่านถูกต้อง'})
            else:
                return jsonify({'success': False, 'message': 'รหัสผ่านไม่ถูกต้อง'})
        else:
            return jsonify({'success': False, 'message': 'ไม่พบสมาชิก'})
            
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/members', methods=['GET'])
def get_members():
    """ดึงข้อมูลสมาชิกทั้งหมด"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        cursor.execute("""
            SELECT m.*, 
                   COALESCE(SUM(s.score), 0) as total_score,
                   COUNT(s.id) as scan_count,
                   MAX(s.scan_time) as last_scan
            FROM members m
            LEFT JOIN scan_logs s ON m.rfid_id = s.rfid_id
            GROUP BY m.id
            ORDER BY total_score DESC, scan_count DESC
        """)
        
        members = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'members': [dict(member) for member in members]
        })
        
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """ตารางคะแนน"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        cursor.execute("""
            SELECT m.username, m.full_name,
                   COALESCE(SUM(s.score), 0) as total_score,
                   COUNT(s.id) as scan_count
            FROM members m
            LEFT JOIN scan_logs s ON m.rfid_id = s.rfid_id
            GROUP BY m.id, m.username, m.full_name
            ORDER BY total_score DESC, scan_count DESC
            LIMIT 20
        """)
        
        leaderboard = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'leaderboard': [dict(row) for row in leaderboard]
        })
        
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/member/<rfid_id>/history', methods=['GET'])
def get_member_history(rfid_id):
    """ประวัติการสแกนของสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        cursor.execute("""
            SELECT bottle_count, can_count, cap_count, label_count, 
                   score, scan_time, image_path
            FROM scan_logs 
            WHERE rfid_id = %s 
            ORDER BY scan_time DESC 
            LIMIT 50
        """, (rfid_id,))
        
        history = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'history': [dict(record) for record in history]
        })
        
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

if __name__ == '__main__':
    print("🚀 PET Detect Member System - PostgreSQL Version")
    print("=" * 50)
    print("Available Routes:")
    print("   - GET  /                    - หน้าเข้าสู่ระบบ")
    print("   - GET  /dashboard           - หน้า Dashboard")
    print("   - GET  /register            - หน้าสมัครสมาชิก")
    print("   - POST /register            - สมัครสมาชิก")
    print("   - GET  /member/<rfid_id>    - รายละเอียดสมาชิก")
    print("   - POST /api/add_score       - เพิ่มคะแนน")
    print("   - GET  /api/members         - ข้อมูลสมาชิกทั้งหมด")
    print("   - GET  /api/leaderboard     - ตารางคะแนน")
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
