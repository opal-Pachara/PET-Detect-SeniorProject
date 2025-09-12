#!/usr/bin/env python3
"""
ระบบสมาชิกสำหรับ PET Detect (SQLite Version)
ใช้ RFID ID เป็น username + เชื่อมกับ SQLite database
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'pet_detect_secret_key_2025'

# Database Configuration
DB_FILE = 'pet_detect_members.db'

def get_db_connection():
    """เชื่อมต่อฐานข้อมูล SQLite"""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_database():
    """สร้างฐานข้อมูลและตาราง"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # ตารางสมาชิก
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rfid_id TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                total_score INTEGER DEFAULT 0,
                scan_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ตารางประวัติการสแกน
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER NOT NULL,
                rfid_id TEXT NOT NULL,
                bottle_count INTEGER DEFAULT 0,
                can_count INTEGER DEFAULT 0,
                cap_count INTEGER DEFAULT 0,
                label_count INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0,
                image_path TEXT,
                scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (member_id) REFERENCES members(id) ON DELETE CASCADE
            )
        ''')
        
        # ตารางการตั้งค่า
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ใส่ค่าเริ่มต้น
        settings = [
            ('bottle_score', '50', 'คะแนนสำหรับขวด'),
            ('can_score', '100', 'คะแนนสำหรับกระป๋อง'),
            ('cap_penalty', '-10', 'หักคะแนนสำหรับฝา'),
            ('label_penalty', '-10', 'หักคะแนนสำหรับสลาก'),
            ('system_name', 'PET Detect Score System', 'ชื่อระบบ'),
            ('auto_register', 'true', 'สมัครสมาชิกอัตโนมัติเมื่อสแกน RFID ใหม่')
        ]
        
        for key, value, desc in settings:
            cursor.execute('''
                INSERT OR IGNORE INTO system_settings (setting_key, setting_value, description) 
                VALUES (?, ?, ?)
            ''', (key, value, desc))
        
        conn.commit()
        conn.close()
        
        logger.info("SQLite database and tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def register_member(rfid_id, full_name=None, email=None, phone=None):
    """สมัครสมาชิกใหม่"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Database connection failed"
        
        cursor = conn.cursor()
        
        # เช็คว่ามีสมาชิกนี้แล้วหรือไม่
        cursor.execute("SELECT id FROM members WHERE rfid_id = ?", (rfid_id,))
        if cursor.fetchone():
            conn.close()
            return False, "RFID ID already registered"
        
        # สร้าง username จาก RFID ID
        username = f"user_{rfid_id[:8]}"
        
        # เพิ่มสมาชิกใหม่
        cursor.execute('''
            INSERT INTO members (rfid_id, username, full_name, email, phone)
            VALUES (?, ?, ?, ?, ?)
        ''', (rfid_id, username, full_name, email, phone))
        
        member_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"New member registered: {rfid_id} -> {username}")
        return True, {"member_id": member_id, "username": username}
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False, str(e)

def get_member_by_rfid(rfid_id):
    """หาสมาชิกจาก RFID ID"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        member = cursor.fetchone()
        
        conn.close()
        
        if member:
            return dict(member)
        return None
        
    except Exception as e:
        logger.error(f"Get member error: {e}")
        return None

def update_member_score(rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path=None):
    """อัพเดทคะแนนสมาชิก"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # หาสมาชิก
        cursor.execute("SELECT id, total_score, scan_count FROM members WHERE rfid_id = ?", (rfid_id,))
        member = cursor.fetchone()
        
        if not member:
            # สมัครสมาชิกอัตโนมัติ
            success, result = register_member(rfid_id)
            if not success:
                conn.close()
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
            SET total_score = ?, scan_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE rfid_id = ?
        ''', (new_score, new_count, rfid_id))
        
        # เพิ่มประวัติการสแกน
        cursor.execute('''
            INSERT INTO scan_logs (member_id, rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (member_id, rfid_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Score updated: {rfid_id}, +{score} (total: {new_score})")
        return True
        
    except Exception as e:
        logger.error(f"Update score error: {e}")
        return False

@app.route('/')
def index():
    """หน้าแรก - ตารางคะแนน"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count, 
                   created_at, updated_at
            FROM members 
            WHERE status = 'active'
            ORDER BY total_score DESC, updated_at DESC
        ''')
        
        members = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return render_template('members.html', members=members)
        
    except Exception as e:
        logger.error(f"Index page error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/member/<rfid_id>')
def member_detail(rfid_id):
    """รายละเอียดสมาชิก"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # ข้อมูลสมาชิก
        cursor.execute("SELECT * FROM members WHERE rfid_id = ?", (rfid_id,))
        member = cursor.fetchone()
        
        if not member:
            return jsonify({'error': 'Member not found'}), 404
        
        member = dict(member)
        
        # ประวัติการสแกน
        cursor.execute('''
            SELECT bottle_count, can_count, cap_count, label_count, 
                   score, scan_timestamp, image_path
            FROM scan_logs 
            WHERE rfid_id = ? 
            ORDER BY scan_timestamp DESC
            LIMIT 50
        ''', (rfid_id,))
        
        scan_history = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return render_template('member_detail.html', member=member, history=scan_history)
        
    except Exception as e:
        logger.error(f"Member detail error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    """หน้าสมัครสมาชิก"""
    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            
            rfid_id = data.get('rfid_id')
            full_name = data.get('full_name')
            email = data.get('email')
            phone = data.get('phone')
            
            if not rfid_id:
                return jsonify({'success': False, 'message': 'RFID ID required'}), 400
            
            success, result = register_member(rfid_id, full_name, email, phone)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Registration successful',
                    'member': result
                })
            else:
                return jsonify({'success': False, 'message': result}), 400
                
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return render_template('register.html')

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
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count, 
                   created_at, updated_at
            FROM members 
            WHERE status = 'active'
            ORDER BY total_score DESC
        ''')
        
        members = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'success': True,
            'total_members': len(members),
            'members': members
        })
        
    except Exception as e:
        logger.error(f"Get members API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard')
def leaderboard():
    """API ตารางคะแนน"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT rfid_id, username, full_name, total_score, scan_count
            FROM members 
            WHERE status = 'active' AND total_score > 0
            ORDER BY total_score DESC
            LIMIT 20
        ''')
        
        members = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # เพิ่มอันดับ
        leaderboard = []
        for i, member in enumerate(members, 1):
            member['rank'] = i
            leaderboard.append(member)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard
        })
        
    except Exception as e:
        logger.error(f"Leaderboard API error: {e}")
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    print("Starting PET Detect Member System (SQLite)...")
    print("Database: SQLite (pet_detect_members.db)")
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
        print("SQLite Database ready!")
        app.run(host='0.0.0.0', port=9000, debug=False)
    else:
        print("Database initialization failed!")
