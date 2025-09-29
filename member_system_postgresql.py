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
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # ตรวจสอบว่ามี admin หรือไม่
        cursor.execute("SELECT id FROM members WHERE username = 'admin'")
        admin_exists = cursor.fetchone()
        
        if not admin_exists:
            # สร้าง admin user
            admin_password = hash_password('admin123')
            cursor.execute("""
                INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                VALUES ('ADMIN001', 'admin', %s, 'System Administrator', 'admin@petdetect.com', '000-000-0000')
            """, (admin_password,))
            print("Admin user created: admin / admin123")
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
        
    except psycopg2.Error as e:
        print(f"Create admin user error: {e}")
        return False

@app.route('/')
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
            # SQLite
            cursor = connection.cursor()
            # สร้าง Row factory สำหรับ SQLite
            connection.row_factory = sqlite3.Row
        
        # ดึงข้อมูลสมาชิกทั้งหมดเรียงตามคะแนน
        if isinstance(connection, psycopg2.extensions.connection):
            # PostgreSQL syntax
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
        else:
            # SQLite syntax
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
        
        # แปลง SQLite Row objects เป็น dict
        if isinstance(connection, sqlite3.Connection):
            members = [dict(row) for row in members]
        
        # สถิติรวม
        cursor.execute("SELECT COUNT(*) as total_members FROM members")
        total_members_row = cursor.fetchone()
        total_members = total_members_row['total_members'] if isinstance(total_members_row, dict) else total_members_row[0]
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        total_score_row = cursor.fetchone()
        total_score = total_score_row['total_score'] if isinstance(total_score_row, dict) else total_score_row[0]
        
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        total_scans_row = cursor.fetchone()
        total_scans = total_scans_row['total_scans'] if isinstance(total_scans_row, dict) else total_scans_row[0]
        
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

@app.route('/admin/login')
def admin_login():
    """หน้าล็อกอินผู้ดูแลระบบ"""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin'))
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login_post():
    """ตรวจสอบการล็อกอินผู้ดูแลระบบ"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        flash('กรุณากรอกชื่อผู้ใช้และรหัสผ่าน', 'error')
        return redirect(url_for('admin_login'))
    
    try:
        connection = get_db_connection()
        if not connection:
            flash('ไม่สามารถเชื่อมต่อฐานข้อมูลได้', 'error')
            return redirect(url_for('admin_login'))
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM members WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if user and verify_password(password, user['password_hash']):
            session['admin_logged_in'] = True
            session['admin_username'] = user['username']
            session['admin_id'] = user['id']
            flash('เข้าสู่ระบบสำเร็จ', 'success')
            return redirect(url_for('admin'))
        else:
            flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'error')
            return redirect(url_for('admin_login'))
            
    except psycopg2.Error as e:
        flash('เกิดข้อผิดพลาดในการเข้าสู่ระบบ', 'error')
        return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin():
    """หน้าผู้ดูแลระบบ - ต้องล็อกอิน"""
    return render_template('admin.html')

@app.route('/admin/logout')
def admin_logout():
    """ออกจากระบบ"""
    session.clear()
    flash('ออกจากระบบเรียบร้อย', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard"""
    return render_template('members.html')

@app.route('/members')
def members():
    """หน้าอันดับ - redirect ไปหน้าหลัก"""
    return redirect(url_for('index'))

@app.route('/register')
def register_page():
    """หน้าสมัครสมาชิก - เริ่มต้นด้วย RFID"""
    return render_template('register.html')

@app.route('/register/rfid')
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
        
        # ตรวจสอบว่า RFID ID มีอยู่แล้วหรือไม่
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
            # ยังไม่ใช่สมาชิก -> ไปหน้าสมัครสมาชิกใหม่
            return jsonify({
                'success': True,
                'action': 'register_new',
                'rfid_id': rfid_id,
                'message': 'กรุณาสมัครสมาชิกใหม่'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/register/new', methods=['POST'])
def register_new_member():
    """สมัครสมาชิกใหม่"""
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

@app.route('/register/password', methods=['POST'])
def register_create_password():
    """สร้างรหัสผ่านสำหรับสมาชิกที่มีอยู่"""
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

@app.route('/register/login', methods=['POST'])
def register_login():
    """ล็อกอินสมาชิก"""
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

@app.route('/member/<rfid_id>')
def member_detail(rfid_id):
    """รายละเอียดสมาชิก"""
    return render_template('member_detail.html', rfid_id=rfid_id)

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
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # เพิ่มข้อมูลการสแกน (รองรับทั้ง PostgreSQL และ SQLite)
        if 'sqlite' in str(type(connection)).lower():
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
        
        return jsonify({'success': True, 'message': 'บันทึกคะแนนสำเร็จ'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/check_member')
def check_member():
    """ตรวจสอบว่าสมาชิกมีอยู่หรือไม่"""
    try:
        rfid_id = request.args.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if member:
            has_password = bool(member['password_hash'])
            return jsonify({
                'success': True,
                'is_member': True,
                'has_password': has_password,
                'member': dict(member)
            })
        else:
            return jsonify({
                'success': True,
                'is_member': False,
                'has_password': False
            })
            
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

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
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if member and verify_password(password, member['password_hash']):
            return jsonify({'success': True, 'message': 'รหัสผ่านถูกต้อง'})
        else:
            return jsonify({'success': False, 'message': 'รหัสผ่านไม่ถูกต้อง'})
            
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/members')
def get_members():
    """ดึงข้อมูลสมาชิกทั้งหมด"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
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

@app.route('/api/leaderboard')
def get_leaderboard():
    """ตารางคะแนน"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
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

@app.route('/api/member/<rfid_id>/history')
def get_member_history(rfid_id):
    """ประวัติการสแกนของสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
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
