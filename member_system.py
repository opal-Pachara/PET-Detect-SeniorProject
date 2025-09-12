#!/usr/bin/env python3
"""
ระบบสมาชิกสำหรับ PET Detect
ใช้ RFID ID เป็น username + เชื่อมกับ database จริง
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import bcrypt
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'pet_detect_secret_key_2025'  # เปลี่ยนใน production

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'pet_detect_db',
    'user': 'pet_user',
    'password': 'pet_password123',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """เชื่อมต่อฐานข้อมูล MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_database():
    """สร้างฐานข้อมูลและตาราง"""
    try:
        # เชื่อมต่อ MySQL server (ไม่ระบุ database)
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database')
        
        connection = mysql.connector.connect(**temp_config)
        cursor = connection.cursor()
        
        # สร้าง database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # ตารางสมาชิก
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS members (
                id INT AUTO_INCREMENT PRIMARY KEY,
                rfid_id VARCHAR(50) UNIQUE NOT NULL,
                username VARCHAR(100) NOT NULL,
                full_name VARCHAR(200),
                email VARCHAR(200),
                phone VARCHAR(20),
                total_score INT DEFAULT 0,
                scan_count INT DEFAULT 0,
                status ENUM('active', 'inactive') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_rfid (rfid_id),
                INDEX idx_username (username)
            )
        ''')
        
        # ตารางประวัติการสแกน
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                member_id INT NOT NULL,
                rfid_id VARCHAR(50) NOT NULL,
                bottle_count INT DEFAULT 0,
                can_count INT DEFAULT 0,
                cap_count INT DEFAULT 0,
                label_count INT DEFAULT 0,
                score INT DEFAULT 0,
                image_path VARCHAR(500),
                scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (member_id) REFERENCES members(id) ON DELETE CASCADE,
                INDEX idx_member (member_id),
                INDEX idx_timestamp (scan_timestamp)
            )
        ''')
        
        # ตารางการตั้งค่า
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                setting_key VARCHAR(100) UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')
        
        # ใส่ค่าเริ่มต้น
        cursor.execute('''
            INSERT IGNORE INTO system_settings (setting_key, setting_value, description) VALUES
            ('bottle_score', '50', 'คะแนนสำหรับขวด'),
            ('can_score', '100', 'คะแนนสำหรับกระป๋อง'),
            ('cap_penalty', '-10', 'หักคะแนนสำหรับฝา'),
            ('label_penalty', '-10', 'หักคะแนนสำหรับสลาก')
        ''')
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info("Database and tables created successfully")
        return True
        
    except Error as e:
        logger.error(f"Database initialization error: {e}")
        return False

def register_member(rfid_id, password=None, full_name=None, email=None, phone=None):
    """สมัครสมาชิกใหม่หรืออัพเดทข้อมูล"""
    try:
        connection = get_db_connection()
        if not connection:
            return False, "Database connection failed"
        
        cursor = connection.cursor()
        
        # เช็คว่ามีสมาชิกนี้แล้วหรือไม่
        cursor.execute("SELECT id, username FROM members WHERE rfid_id = %s", (rfid_id,))
        existing_member = cursor.fetchone()
        
        if existing_member:
            # มีสมาชิกอยู่แล้ว - อัพเดทข้อมูล
            member_id, username = existing_member
            
            # Hash password ถ้ามี
            password_hash = None
            if password:
                import bcrypt
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # อัพเดทข้อมูล
            update_query = '''
                UPDATE members 
                SET full_name = COALESCE(%s, full_name),
                    email = COALESCE(%s, email),
                    phone = COALESCE(%s, phone),
                    password_hash = COALESCE(%s, password_hash),
                    updated_at = CURRENT_TIMESTAMP
                WHERE rfid_id = %s
            '''
            cursor.execute(update_query, (full_name, email, phone, password_hash, rfid_id))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info(f"Member updated: {rfid_id} -> {username}")
            return True, {"member_id": member_id, "username": username, "action": "updated"}
        else:
            # สร้างสมาชิกใหม่
            username = f"user_{rfid_id[:8]}"
            
            # Hash password ถ้ามี
            password_hash = None
            if password:
                import bcrypt
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # เพิ่มสมาชิกใหม่
            insert_query = '''
                INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                VALUES (%s, %s, %s, %s, %s, %s)
            '''
            cursor.execute(insert_query, (rfid_id, username, password_hash, full_name, email, phone))
            
            member_id = cursor.lastrowid
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info(f"New member registered: {rfid_id} -> {username}")
            return True, {"member_id": member_id, "username": username, "action": "created"}
        
    except Error as e:
        logger.error(f"Registration error: {e}")
        return False, str(e)

def get_member_by_rfid(rfid_id):
    """หาสมาชิกจาก RFID ID"""
    try:
        connection = get_db_connection()
        if not connection:
            return None
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return member
        
    except Error as e:
        logger.error(f"Get member error: {e}")
        return None

def update_member_info(rfid_id, username, password, full_name, email, phone):
    """อัพเดทข้อมูลสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            logger.error("Database connection failed")
            return False
        
        cursor = connection.cursor()
        
        # Hash password
        password_hash = None
        if password:
            try:
                import bcrypt
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except Exception as e:
                logger.error(f"Password hashing error: {e}")
                return False
        
        # อัพเดทข้อมูล
        update_query = '''
            UPDATE members 
            SET username = %s,
                password_hash = %s,
                full_name = %s,
                email = %s,
                phone = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE rfid_id = %s
        '''
        
        logger.info(f"Updating member: {rfid_id} -> {username}")
        cursor.execute(update_query, (username, password_hash, full_name, email, phone, rfid_id))
        
        if cursor.rowcount == 0:
            logger.error(f"No member found with RFID ID: {rfid_id}")
            cursor.close()
            connection.close()
            return False
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Member updated successfully: {rfid_id} -> {username}")
        return True
        
    except Error as e:
        logger.error(f"Update member error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def update_member_score(rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path=None):
    """อัพเดทคะแนนสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # หาสมาชิก
        cursor.execute("SELECT id, total_score, scan_count FROM members WHERE rfid_id = %s", (rfid_id,))
        member = cursor.fetchone()
        
        if not member:
            # สมัครสมาชิกอัตโนมัติ
            success, result = register_member(rfid_id)
            if not success:
                cursor.close()
                connection.close()
                return False
            member_id = result["member_id"]
            old_score = 0
            old_count = 0
        else:
            member_id, old_score, old_count = member
        
        # อัพเดทคะแนน
        new_score = old_score + score
        new_count = old_count + 1
        
        cursor.execute('''
            UPDATE members 
            SET total_score = %s, scan_count = %s, updated_at = CURRENT_TIMESTAMP
            WHERE rfid_id = %s
        ''', (new_score, new_count, rfid_id))
        
        # เพิ่มประวัติการสแกน
        cursor.execute('''
            INSERT INTO scan_logs (member_id, rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (member_id, rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        logger.info(f"Score updated: {rfid_id}, +{score} (total: {new_score})")
        return True
        
    except Error as e:
        logger.error(f"Update score error: {e}")
        return False

@app.route('/')
def index():
    """หน้าแรก - หน้า Login"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard - ตารางคะแนน"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count, 
                   created_at, updated_at
            FROM members 
            WHERE status = 'active'
            ORDER BY total_score DESC, updated_at DESC
        ''')
        
        members = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return render_template('members.html', members=members)
        
    except Error as e:
        logger.error(f"Index page error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['GET', 'POST'])
def register():
    """หน้าสมัครสมาชิก"""
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            rfid_id = data.get('rfid_id')
            password = data.get('password')
            full_name = data.get('full_name')
            email = data.get('email')
            phone = data.get('phone')
            
            if not rfid_id:
                return jsonify({'success': False, 'message': 'RFID ID required'}), 400
            
            success, result = register_member(rfid_id, password, full_name, email, phone)
            
            if success:
                action = result.get('action', 'created')
                message = 'Registration successful' if action == 'created' else 'Member information updated'
                return jsonify({
                    'success': True,
                    'message': message,
                    'action': action,
                    'member': result
                })
            else:
                return jsonify({'success': False, 'message': result}), 400
                
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return render_template('register.html')

@app.route('/members')
def members():
    """หน้าแสดงอันดับสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # ดึงข้อมูลสมาชิกทั้งหมดเรียงตามคะแนน
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count, 
                   created_at, updated_at, status
            FROM members 
            WHERE status = 'active'
            ORDER BY total_score DESC, scan_count DESC
        ''')
        
        members = cursor.fetchall()
        
        # สถิติรวม
        cursor.execute('SELECT COUNT(*) as total_members FROM members WHERE status = "active"')
        total_members = cursor.fetchone()['total_members']
        
        cursor.execute('SELECT SUM(total_score) as total_score FROM members WHERE status = "active"')
        total_score = cursor.fetchone()['total_score'] or 0
        
        cursor.execute('SELECT SUM(scan_count) as total_scans FROM members WHERE status = "active"')
        total_scans = cursor.fetchone()['total_scans'] or 0
        
        cursor.close()
        connection.close()
        
        return render_template('members.html', 
                             members=members, 
                             total_members=total_members,
                             total_score=total_score,
                             total_scans=total_scans)
        
    except Exception as e:
        logger.error(f"Members page error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/member_login')
def member_login():
    """หน้าเข้าสู่ระบบสมาชิก"""
    return render_template('member_login.html')

@app.route('/member_detail/<rfid_id>')
def member_detail(rfid_id):
    """หน้ารายละเอียดสมาชิก"""
    return render_template('member_detail.html', rfid_id=rfid_id)

@app.route('/edit_profile')
def edit_profile():
    """หน้าแก้ไขข้อมูลสมาชิก"""
    return render_template('edit_profile.html')

@app.route('/api/add_score', methods=['POST'])
def add_score():
    """API สำหรับเพิ่มคะแนน (เรียกจาก Pi)"""
    try:
        data = request.get_json()
        
        rfid_id = data.get('card_id') or data.get('rfid_id')
        bottle_count = data.get('bottle_count', 0)
        can_count = data.get('can_count', 0)
        cap_count = data.get('cap_count', 0)
        label_count = data.get('label_count', 0)
        score = data.get('score', 0)
        image_path = data.get('image_path')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'RFID ID required'}), 400
        
        success = update_member_score(
            rfid_id=rfid_id,
            bottle_count=bottle_count,
            can_count=can_count,
            cap_count=cap_count,
            label_count=label_count,
            score=score,
            image_path=image_path
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Score added successfully',
                'rfid_id': rfid_id,
                'score': score
            })
        else:
            return jsonify({'success': False, 'message': 'Database error'}), 500
            
    except Exception as e:
        logger.error(f"Add score API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/members')
def get_all_members():
    """API ดึงข้อมูลสมาชิกทั้งหมด"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count, 
                   created_at, updated_at
            FROM members 
            WHERE status = 'active'
            ORDER BY total_score DESC
        ''')
        
        members = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'total_members': len(members),
            'members': members
        })
        
    except Error as e:
        logger.error(f"Get members API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard')
def leaderboard():
    """API ตารางคะแนน"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count
            FROM members 
            WHERE status = 'active' AND total_score > 0
            ORDER BY total_score DESC
            LIMIT 20
        ''')
        
        members = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # เพิ่มอันดับ
        leaderboard = []
        for i, member in enumerate(members, 1):
            member['rank'] = i
            leaderboard.append(member)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard
        })
        
    except Error as e:
        logger.error(f"Leaderboard API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_member', methods=['POST'])
def check_member():
    """API ตรวจสอบว่าสมาชิกมีอยู่หรือไม่ และมี password หรือไม่"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'RFID ID required'}), 400
        
        member = get_member_by_rfid(rfid_id)
        
        if member:
            # ตรวจสอบว่ามี password หรือไม่
            has_password = member.get('password_hash') is not None and member.get('password_hash') != ''
            
            return jsonify({
                'success': True,
                'is_member': True,
                'has_password': has_password,
                'member': member
            })
        else:
            return jsonify({
                'success': True,
                'is_member': False,
                'has_password': False,
                'message': 'Not a member yet'
            })
            
    except Exception as e:
        logger.error(f"Check member API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/verify_password', methods=['POST'])
def verify_password():
    """API ตรวจสอบรหัสผ่าน"""
    try:
        data = request.get_json()
        rfid_id = data.get('rfid_id')
        password = data.get('password')
        
        if not rfid_id or not password:
            return jsonify({'success': False, 'message': 'RFID ID and password required'}), 400
        
        member = get_member_by_rfid(rfid_id)
        
        if not member:
            return jsonify({'success': False, 'message': 'Member not found'}), 404
        
        if not member.get('password_hash'):
            return jsonify({'success': False, 'message': 'No password set'}), 400
        
        # ตรวจสอบรหัสผ่าน
        import bcrypt
        password_valid = bcrypt.checkpw(password.encode('utf-8'), member['password_hash'].encode('utf-8'))
        
        return jsonify({
            'success': True,
            'valid': password_valid,
            'message': 'Password verified' if password_valid else 'Invalid password'
        })
        
    except Exception as e:
        logger.error(f"Verify password API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_member', methods=['POST'])
def update_member():
    """API อัพเดทข้อมูลสมาชิก"""
    try:
        data = request.get_json()
        
        rfid_id = data.get('rfid_id')
        username = data.get('username')
        password = data.get('password')
        full_name = data.get('full_name')
        email = data.get('email')
        phone = data.get('phone')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'RFID ID required'}), 400
        
        if not username:
            return jsonify({'success': False, 'message': 'Username required'}), 400
        
        # อัพเดทข้อมูลสมาชิก
        success = update_member_info(rfid_id, username, password, full_name, email, phone)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Member information updated successfully'
            })
        else:
            return jsonify({'success': False, 'message': 'Update failed'}), 500
            
    except Exception as e:
        logger.error(f"Update member API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/member/<rfid_id>')
def get_member(rfid_id):
    """API ดึงข้อมูลสมาชิกคนเดียว"""
    try:
        member = get_member_by_rfid(rfid_id)
        
        if member:
            return jsonify({
                'success': True,
                'member': member
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Member not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Get member API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/member/<rfid_id>/history')
def get_member_history(rfid_id):
    """API ประวัติการสแกนของสมาชิก"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # ประวัติการสแกน
        cursor.execute('''
            SELECT bottle_count, can_count, cap_count, label_count, 
                   score, scan_timestamp as scan_time, image_path
            FROM scan_logs 
            WHERE rfid_id = %s 
            ORDER BY scan_timestamp DESC
            LIMIT 50
        ''', (rfid_id,))
        
        history = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'history': history
        })
        
    except Exception as e:
        logger.error(f"Get member history error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print("Starting PET Detect Member System...")
    print("Database: MySQL")
    print("URL: http://localhost:9000")
    print("API Endpoints:")
    print("   - GET  /                    - หน้าแรก (ตารางคะแนน)")
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
        app.run(host='0.0.0.0', port=9000, debug=False)
    else:
        print("Database initialization failed!")
        print("Please check MySQL server and configuration")
