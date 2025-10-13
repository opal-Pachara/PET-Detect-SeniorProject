#!/usr/bin/env python3
"""
PET Detect Member System - SQLite Version for Cloud Deployment
ใช้ SQLite แทน PostgreSQL เพื่อหลีกเลี่ยง Build Issues
"""

import os
import sqlite3
import hashlib
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime

app = Flask(__name__)

# SQLite Database configuration
DATABASE_PATH = os.environ.get('DATABASE_URL', 'pet_detect_members.db')

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล SQLite"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """สร้างตารางฐานข้อมูล"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # สร้างตาราง members
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rfid_id TEXT UNIQUE NOT NULL,
                username TEXT,
                password_hash TEXT,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # สร้างตาราง scan_logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rfid_id TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                image_path TEXT,
                scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (rfid_id) REFERENCES members(rfid_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        print("Database tables created successfully!")
        return True
        
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        return False

def hash_password(password):
    """เข้ารหัสรหัสผ่าน"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """ตรวจสอบรหัสผ่าน"""
    return hash_password(password) == hashed

@app.route('/')
def index():
    """หน้าหลัก"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """หน้า Dashboard"""
    return render_template('members.html')

@app.route('/register')
def register_page():
    """หน้าสมัครสมาชิก"""
    rfid_id = request.args.get('rfid_id', '')
    return render_template('register.html', rfid_id=rfid_id)

@app.route('/register', methods=['POST'])
def register_member():
    """สมัครสมาชิก"""
    try:
        rfid_id = request.form.get('rfid_id')
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        
        if not all([rfid_id, username, password, full_name, phone]):
            return jsonify({'success': False, 'message': 'กรุณากรอกข้อมูลให้ครบถ้วน'})
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
        
        # ตรวจสอบว่า RFID ID มีอยู่แล้วหรือไม่
        cursor.execute("SELECT id FROM members WHERE rfid_id = ?", (rfid_id,))
        existing_member = cursor.fetchone()
        
        password_hash = hash_password(password)
        
        if existing_member:
            # อัพเดทข้อมูลสมาชิกที่มีอยู่
            cursor.execute("""
                UPDATE members 
                SET username = ?, password_hash = ?, full_name = ?, 
                    email = ?, phone = ?, updated_at = CURRENT_TIMESTAMP
                WHERE rfid_id = ?
            """, (username, password_hash, full_name, email, phone, rfid_id))
        else:
            # สร้างสมาชิกใหม่
            cursor.execute("""
                INSERT INTO members (rfid_id, username, password_hash, full_name, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (rfid_id, username, password_hash, full_name, email, phone))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'สมัครสมาชิกสำเร็จ'})
        
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/member/<rfid_id>')
def member_detail(rfid_id):
    """รายละเอียดสมาชิก"""
    return render_template('member_detail.html', rfid_id=rfid_id)

@app.route('/api/add_score', methods=['POST'])
def add_score():
    """เพิ่มคะแนนจาก RFID scan"""
    try:
        data = request.json
        rfid_id = data.get('rfid_id')
        score = data.get('score', 0)
        image_path = data.get('image_path', '')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
        
        # เพิ่มข้อมูลการสแกน
        cursor.execute("""
            INSERT INTO scan_logs (rfid_id, score, image_path)
            VALUES (?, ?, ?)
        """, (rfid_id, score, image_path))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'บันทึกคะแนนสำเร็จ'})
        
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/check_member')
def check_member():
    """ตรวจสอบว่าสมาชิกมีอยู่หรือไม่"""
    try:
        rfid_id = request.args.get('rfid_id')
        
        if not rfid_id:
            return jsonify({'success': False, 'message': 'ไม่พบ RFID ID'})
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        member = cursor.fetchone()
        
        conn.close()
        
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
            
    except sqlite3.Error as e:
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
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        member = cursor.fetchone()
        
        conn.close()
        
        if member and verify_password(password, member['password_hash']):
            return jsonify({'success': True, 'message': 'รหัสผ่านถูกต้อง'})
        else:
            return jsonify({'success': False, 'message': 'รหัสผ่านไม่ถูกต้อง'})
            
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/members')
def get_members():
    """ดึงข้อมูลสมาชิกทั้งหมด"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
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
        
        conn.close()
        
        return jsonify({
            'success': True,
            'members': [dict(member) for member in members]
        })
        
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/leaderboard')
def get_leaderboard():
    """ตารางคะแนน"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
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
        
        conn.close()
        
        return jsonify({
            'success': True,
            'leaderboard': [dict(row) for row in leaderboard]
        })
        
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/member/<rfid_id>/history')
def get_member_history(rfid_id):
    """ประวัติการสแกนของสมาชิก"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้'})
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM scan_logs 
            WHERE rfid_id = ? 
            ORDER BY scan_time DESC 
            LIMIT 50
        """, (rfid_id,))
        
        history = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'history': [dict(record) for record in history]
        })
        
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

if __name__ == '__main__':
    print("🚀 PET Detect Member System - SQLite Version")
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
        # Get port from environment variable (for cloud deployment)
        port = int(os.environ.get('PORT', 9000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Database initialization failed!")
        print("Please check SQLite configuration")
