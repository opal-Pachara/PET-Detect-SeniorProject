#!/usr/bin/env python3
"""
PostgreSQL Render

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

# Database config
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'pet_detect_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'password')
}


print(f"Database Config: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']} (user: {DB_CONFIG['user']})")

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล PostgreSQL เเละ SQLite fallback"""
    try:
        # ลองเชื่อม PostgreSQL ก่อน
        connection = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL")
        return connection
    except psycopg2.Error as e:
        print(f"PostgreSQL connection failed: {e}")
        try:
            # Fallback to SQLite
            connection = sqlite3.connect('pet_detect.db')
            
            connection.row_factory = sqlite3.Row
            print("Connected to SQLite (fallback)")
            return connection
        except Exception as sqlite_error:
            print(f"SQLite connection also failed: {sqlite_error}")
            return None

def init_database():
    """สร้างตารางถ้ายังไม่มี"""
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # ตรวจสอบว่าเป็น PostgreSQL หรือ SQLite
        if isinstance(connection, psycopg2.extensions.connection):
            # PostgreSQL ตาราง scan_logs และ admin_users
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
        else:
            # SQLite ตาราง scan_logs และ admin_users
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
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS member_names (
                    rfid_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL
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
    """การเข้ารหัส"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """ตรวจสอบรหัส"""
    return hash_password(password) == hashed

def login_required(f):
    """ตรวจสอบการล็อกอิน"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('กรุณาล็อกอินก่อนเข้าสู่ระบบ', 'warning')
            return redirect(url_for('admin_login'))
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
        
        # ตรวจสอบว่ามี admin ไหม
        if isinstance(connection, psycopg2.extensions.connection):
            print("DEBUG: Checking admin user with PostgreSQL")
            cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        else:
            print("DEBUG: Checking admin user with SQLite")
            cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        if not admin_exists:
            # สร้าง admin
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    INSERT INTO admin_users (username, password_hash)
                    VALUES ('admin', %s)
                """, (admin_password,))
            else:
                cursor.execute("""
                    INSERT INTO admin_users (username, password_hash)
                    VALUES ('admin', ?)
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
    """หน้าหลัก ตารางคะแนน"""
    try:
        # สร้างตารางถ้าไม่มี
        init_database()
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            
            cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิกทั้งหมดเรียงตามคะแนน รวมชื่อจาก member_names
        try:
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    SELECT 
                        s.rfid_id,
                        m.display_name,
                        SUM(s.score) as total_score,
                        COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                        MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    LEFT JOIN member_names m ON s.rfid_id = m.rfid_id
                    GROUP BY s.rfid_id, m.display_name
                    ORDER BY total_score DESC, scan_count DESC
                """)
            else:
                cursor.execute("""
                    SELECT 
                        s.rfid_id,
                        m.display_name,
                        SUM(s.score) as total_score,
                        COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                        MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    LEFT JOIN member_names m ON s.rfid_id = m.rfid_id
                    GROUP BY s.rfid_id, m.display_name
                    ORDER BY total_score DESC, scan_count DESC
                """)
        except Exception as e:
            print(f"Member names query fallback: {e}")
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    SELECT rfid_id, NULL as display_name, SUM(score) as total_score,
                        COUNT(CASE WHEN image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                        MAX(scan_time) as last_scan
                    FROM scan_logs
                    GROUP BY rfid_id
                    ORDER BY total_score DESC, scan_count DESC
                """)
            else:
                cursor.execute("""
                    SELECT rfid_id, NULL as display_name, SUM(score) as total_score,
                        COUNT(CASE WHEN image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                        MAX(scan_time) as last_scan
                    FROM scan_logs
                    GROUP BY rfid_id
                    ORDER BY total_score DESC, scan_count DESC
                """)
        
        members = cursor.fetchall()
        
        if isinstance(connection, sqlite3.Connection):
            members = [dict(row) for row in members]
        else:
            members = [dict(row) for row in members]
        
        # สร้าง display_label: "ชื่อ RFID" หรือแค่ rfid_id
        for m in members:
            dn = m.get('display_name')
            m['display_label'] = f"{dn} {m['rfid_id']}" if dn else m['rfid_id']
        
        # ตรวจสอบข้อมูลที่ได้
        print(f"DEBUG: Found {len(members)} members")
        for i, member in enumerate(members[:3]):  # แสดงแค่ 3 คนแรก
            print(f"DEBUG: Member {i+1}: {member}")
        
        # ตรวจสอบ raw data
        print(f"DEBUG: Raw members data: {members}")
        
        # สถิติรวม
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_members FROM scan_logs")
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
            cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = %s", (username,))
        else:
            print("DEBUG: Using SQLite query")
            cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = ?", (username,))
        
        user = cursor.fetchone()
        print(f"DEBUG: User found: {user}")
        
        if user:
            # แปลง user เป็น dict
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
    """หน้าผู้ดูแลระบบ"""
    return render_template('admin.html')

@app.route('/admin/logout', methods=['GET'])
def admin_logout():
    """ออกจากระบบ"""
    session.clear()
    flash('ออกจากระบบเรียบร้อย', 'info')
    return redirect(url_for('admin_login'))

@app.route('/api/debug/admin', methods=['GET'])
def debug_admin():
    """Debug สำหรับตรวจสอบ admin"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ตรวจสอบ admin
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = 'admin'")
        else:
            cursor.execute("SELECT id, username, password_hash FROM admin_users WHERE username = 'admin'")
        
        admin_user = cursor.fetchone()
        
        # ตรวจสอบจำนวนสมาชิกทั้งหมด
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT COUNT(DISTINCT rfid_id) as count FROM scan_logs")
        else:
            cursor.execute("SELECT COUNT(DISTINCT rfid_id) as count FROM scan_logs")
        
        member_count = cursor.fetchone()
        
        # ตรวจสอบ database type ก่อนปิดการเชื่อมต่อ
        is_postgresql = isinstance(connection, psycopg2.extensions.connection)
        
        connection.close()
        
        # แปลงข้อมูลให้เหมาะกับ database type
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
        
        # ตรวจสอบว่ามี admin ไหม
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        else:
            cursor.execute("SELECT id FROM admin_users WHERE username = 'admin'")
        
        admin_exists = cursor.fetchone()
        print(f"DEBUG: Admin exists: {admin_exists}")
        
        if not admin_exists:
            # สร้าง admin
            admin_password = hash_password('admin123')
            print(f"DEBUG: Admin password hash: {admin_password}")
            
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    INSERT INTO admin_users (username, password_hash)
                    VALUES ('admin', %s)
                """, (admin_password,))
            else:
                cursor.execute("""
                    INSERT INTO admin_users (username, password_hash)
                    VALUES ('admin', ?)
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

@app.route('/api/debug_data', methods=['GET'])
def debug_data():
    """API สำหรับดูข้อมูลในฐานข้อมูล"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ตรวจสอบตารางที่มีอยู่
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
        else:
            cursor.execute("""
                SELECT name as table_name 
                FROM sqlite_master 
                WHERE type='table'
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
        
        # แปลงข้อมูลสำหรับ SQLite และ PostgreSQL
        if isinstance(connection, sqlite3.Connection):
            members_data = [dict(row) for row in members_data]
            tables = [dict(row) for row in tables]
        else:
            # PostgreSQL แปลงเป็น list
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
        
        # แปลงข้อมูลสำหรับ SQLite
        if isinstance(connection, sqlite3.Connection):
            recent_scans = [dict(row) for row in recent_scans]
        else:
            # PostgreSQL return เป็น dict
            recent_scans = [dict(scan) for scan in recent_scans]
        
        # สถิติรวม
        cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_users FROM scan_logs")
        total_users_row = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
        total_scans_row = cursor.fetchone()
        
        cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
        total_score_row = cursor.fetchone()
        
        # แปลงข้อมูลสำหรับ SQLite
        if isinstance(connection, sqlite3.Connection):
            total_users = total_users_row['total_users'] if total_users_row else 0
            total_scans = total_scans_row['total_scans'] if total_scans_row else 0
            total_score = total_score_row['total_score'] if total_score_row else 0
        else:
            total_users = total_users_row['total_users'] if total_users_row else 0
            total_scans = total_scans_row['total_scans'] if total_scans_row else 0
            total_score = total_score_row['total_score'] if total_score_row else 0
        
        # ข้อมูล admin users
        cursor.execute("SELECT username, created_at FROM admin_users")
        admin_users = cursor.fetchall()
        
        # แปลงข้อมูลสำหรับ SQLite
        if isinstance(connection, sqlite3.Connection):
            admin_users = [dict(row) for row in admin_users]
        else:
            # PostgreSQL return เป็น dict
            admin_users = [dict(user) for user in admin_users]
        
        # ตรวจสอบ database type
        database_type = "PostgreSQL" if isinstance(connection, psycopg2.extensions.connection) else "SQLite"
        
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
        
        # สถิติสมาชิก (scan_logs)
        if isinstance(connection, psycopg2.extensions.connection):
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
        else:
            cursor.execute("SELECT COUNT(DISTINCT rfid_id) as total_members FROM scan_logs")
            result = cursor.fetchone()
            total_members = result['total_members'] if result else 0
            
            cursor.execute("SELECT COUNT(*) as total_scans FROM scan_logs")
            result = cursor.fetchone()
            total_scans = result['total_scans'] if result else 0
            
            cursor.execute("SELECT COALESCE(SUM(score), 0) as total_score FROM scan_logs")
            result = cursor.fetchone()
            total_score = result['total_score'] if result else 0
            
            
            cursor.execute("""
                SELECT COUNT(DISTINCT rfid_id) as active_users 
                FROM scan_logs 
                WHERE scan_time >= datetime('now', '-30 days')
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
    """หน้าจัดการสมาชิก"""
    try:
        init_database()
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # ดึงข้อมูลสมาชิกจาก scan_logs + member_names (ชื่อที่ Admin ตั้ง)
        try:
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    SELECT s.rfid_id, m.display_name,
                           COALESCE(SUM(s.score), 0) as total_score,
                           COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                           MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    LEFT JOIN member_names m ON s.rfid_id = m.rfid_id
                    GROUP BY s.rfid_id, m.display_name
                    ORDER BY total_score DESC, scan_count DESC
                """)
            else:
                cursor.execute("""
                    SELECT s.rfid_id, m.display_name,
                           COALESCE(SUM(s.score), 0) as total_score,
                           COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                           MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    LEFT JOIN member_names m ON s.rfid_id = m.rfid_id
                    GROUP BY s.rfid_id, m.display_name
                    ORDER BY total_score DESC, scan_count DESC
                """)
        except Exception as e:
            print(f"Manage members query fallback: {e}")
            if isinstance(connection, psycopg2.extensions.connection):
                cursor.execute("""
                    SELECT s.rfid_id, NULL as display_name,
                           COALESCE(SUM(s.score), 0) as total_score,
                           COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                           MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    GROUP BY s.rfid_id
                    ORDER BY total_score DESC, scan_count DESC
                """)
            else:
                cursor.execute("""
                    SELECT s.rfid_id, NULL as display_name,
                           COALESCE(SUM(s.score), 0) as total_score,
                           COUNT(CASE WHEN s.image_path != 'ADMIN_ADJUSTMENT' THEN 1 END) as scan_count,
                           MAX(s.scan_time) as last_scan
                    FROM scan_logs s
                    GROUP BY s.rfid_id
                    ORDER BY total_score DESC, scan_count DESC
                """)
        
        members = cursor.fetchall()
        
        # แปลงข้อมูลสำหรับ SQLite และสร้าง display_label (ชื่อ RFID)
        if isinstance(connection, sqlite3.Connection):
            members = [dict(member) for member in members]
        else:
            members = [dict(member) for member in members]
        
        for m in members:
            dn = m.get('display_name')
            m['display_label'] = f"{dn} {m['rfid_id']}" if dn else m['rfid_id']
        
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
                SELECT s.*, s.rfid_id as full_name, s.rfid_id as username
                FROM scan_logs s
                WHERE s.image_path != 'ADMIN_ADJUSTMENT'
                ORDER BY s.scan_time DESC
                LIMIT 100
            """)
        else:
            cursor.execute("""
                SELECT s.*, s.rfid_id as full_name, s.rfid_id as username
                FROM scan_logs s
                WHERE s.image_path != 'ADMIN_ADJUSTMENT'
                ORDER BY s.scan_time DESC
                LIMIT 100
            """)
        
        scan_history = cursor.fetchall()
        
        # แปลงข้อมูลรูปแบบวันที่
        if isinstance(connection, sqlite3.Connection):
            scan_history = [dict(scan) for scan in scan_history]
        
        # จัดรูปเป็น string
        for scan in scan_history:
            if scan.get('scan_time'):
                if isinstance(scan['scan_time'], str):
                    # ถ้าเป็น string แล้ว ให้แปลงเป็น datetime
                    try:
                        from datetime import datetime
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
        
        if isinstance(connection, psycopg2.extensions.connection):
            cursor = connection.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = connection.cursor()
        
        # คำนวณคะแนนที่ต้องปรับ
        query = "SELECT SUM(score) as current_total FROM scan_logs WHERE rfid_id = %s" if isinstance(connection, psycopg2.extensions.connection) else "SELECT SUM(score) as current_total FROM scan_logs WHERE rfid_id = ?"
        cursor.execute(query, (rfid_id,))
        result = cursor.fetchone()
        if isinstance(connection, sqlite3.Connection):
            current_total = result['current_total'] if result and result['current_total'] else 0
        else:
            current_total = result['current_total'] if result and result['current_total'] else 0
        score_difference = new_score - current_total
        
        # เพิ่ม record ใหม่เพื่อปรับคะแนน
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("""
                INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (rfid_id, 0, 0, 0, 0, score_difference, 'ADMIN_ADJUSTMENT'))
        else:
            cursor.execute("""
                INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
    """แก้ไขชื่อสมาชิก (รูปแบบ: ชื่อ RFID)"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        display_name = (data.get('display_name') or '').strip()
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        try:
            if display_name:
                if isinstance(connection, psycopg2.extensions.connection):
                    cursor.execute("""
                        INSERT INTO member_names (rfid_id, display_name)
                        VALUES (%s, %s)
                        ON CONFLICT (rfid_id) DO UPDATE SET display_name = EXCLUDED.display_name
                    """, (rfid_id, display_name))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO member_names (rfid_id, display_name)
                        VALUES (?, ?)
                    """, (rfid_id, display_name))
            else:
                if isinstance(connection, psycopg2.extensions.connection):
                    cursor.execute("DELETE FROM member_names WHERE rfid_id = %s", (rfid_id,))
                else:
                    cursor.execute("DELETE FROM member_names WHERE rfid_id = ?", (rfid_id,))
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
        
        # ลบข้อมูลจาก scan_logs และ member_names
        if isinstance(connection, psycopg2.extensions.connection):
            cursor.execute("DELETE FROM scan_logs WHERE rfid_id = %s", (rfid_id,))
            try:
                cursor.execute("DELETE FROM member_names WHERE rfid_id = %s", (rfid_id,))
            except Exception:
                pass
        else:
            cursor.execute("DELETE FROM scan_logs WHERE rfid_id = ?", (rfid_id,))
            try:
                cursor.execute("DELETE FROM member_names WHERE rfid_id = ?", (rfid_id,))
            except Exception:
                pass
        
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
        
        score = (bottle_count * 50) + (can_count * 100) - (cap_count * 10) - (label_count * 10)
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = connection.cursor()
        
        # เพิ่มข้อมูลการสแกน
        if isinstance(connection, sqlite3.Connection):
            # SQLite
            cursor.execute("""
                INSERT INTO scan_logs (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        else:
            # PostgreSQL
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

