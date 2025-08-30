#!/usr/bin/env python3
"""
เว็บแสดงคะแนนของแต่ละบัตร RFID
ใช้ Flask + SQLite สำหรับเก็บข้อมูล
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database file
DB_FILE = 'rfid_scores.db'

def init_database():
    """สร้างฐานข้อมูล"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # ตาราง RFID cards
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rfid_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT UNIQUE NOT NULL,
            card_name TEXT,
            total_score INTEGER DEFAULT 0,
            scan_count INTEGER DEFAULT 0,
            first_scan TIMESTAMP,
            last_scan TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # ตารางประวัติการสแกน
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT NOT NULL,
            bottle_count INTEGER DEFAULT 0,
            can_count INTEGER DEFAULT 0,
            cap_count INTEGER DEFAULT 0,
            label_count INTEGER DEFAULT 0,
            score INTEGER DEFAULT 0,
            image_path TEXT,
            scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (card_id) REFERENCES rfid_cards (card_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def add_or_update_card(card_id, bottle_count=0, can_count=0, cap_count=0, label_count=0, score=0, image_path=None):
    """เพิ่มหรืออัพเดทคะแนนบัตร"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # เช็คว่ามีบัตรนี้แล้วหรือไม่
        cursor.execute('SELECT total_score, scan_count FROM rfid_cards WHERE card_id = ?', (card_id,))
        result = cursor.fetchone()
        
        if result:
            # อัพเดทบัตรที่มีอยู่
            old_score, old_count = result
            new_score = old_score + score
            new_count = old_count + 1
            
            cursor.execute('''
                UPDATE rfid_cards 
                SET total_score = ?, scan_count = ?, last_scan = CURRENT_TIMESTAMP
                WHERE card_id = ?
            ''', (new_score, new_count, card_id))
            
            logger.info(f"Updated card {card_id}: score {old_score} → {new_score}")
        else:
            # เพิ่มบัตรใหม่
            cursor.execute('''
                INSERT INTO rfid_cards (card_id, total_score, scan_count, first_scan, last_scan)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (card_id, score))
            
            logger.info(f"Added new card {card_id}: score {score}")
        
        # เพิ่มประวัติการสแกน
        cursor.execute('''
            INSERT INTO scan_history (card_id, bottle_count, can_count, cap_count, label_count, score, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (card_id, bottle_count, can_count, cap_count, label_count, score, image_path))
        
        conn.commit()
        return True
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        conn.close()

@app.route('/')
def index():
    """หน้าแรก - แสดงตารางคะแนน"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # ดึงข้อมูลบัตรทั้งหมด เรียงตามคะแนน
    cursor.execute('''
        SELECT card_id, card_name, total_score, scan_count, 
               first_scan, last_scan
        FROM rfid_cards 
        ORDER BY total_score DESC, last_scan DESC
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    # แปลงเป็น list of dict
    card_list = []
    for card in cards:
        card_dict = {
            'card_id': card[0],
            'card_name': card[1] or f"Card-{card[0][:8]}",
            'total_score': card[2],
            'scan_count': card[3],
            'first_scan': card[4],
            'last_scan': card[5]
        }
        card_list.append(card_dict)
    
    return render_template('index.html', cards=card_list)

@app.route('/card/<card_id>')
def card_detail(card_id):
    """รายละเอียดบัตร"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # ข้อมูลบัตร
    cursor.execute('SELECT * FROM rfid_cards WHERE card_id = ?', (card_id,))
    card = cursor.fetchone()
    
    if not card:
        return jsonify({'error': 'Card not found'}), 404
    
    # ประวัติการสแกน
    cursor.execute('''
        SELECT bottle_count, can_count, cap_count, label_count, 
               score, scan_timestamp, image_path
        FROM scan_history 
        WHERE card_id = ? 
        ORDER BY scan_timestamp DESC
    ''', (card_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    card_info = {
        'card_id': card[1],
        'card_name': card[2] or f"Card-{card[1][:8]}",
        'total_score': card[3],
        'scan_count': card[4],
        'first_scan': card[5],
        'last_scan': card[6]
    }
    
    # แปลงประวัติ
    history_list = []
    for h in history:
        history_dict = {
            'bottle_count': h[0],
            'can_count': h[1],
            'cap_count': h[2],
            'label_count': h[3],
            'score': h[4],
            'timestamp': h[5],
            'image_path': h[6]
        }
        history_list.append(history_dict)
    
    return render_template('card_detail.html', card=card_info, history=history_list)

@app.route('/api/add_score', methods=['POST'])
def add_score():
    """API สำหรับเพิ่มคะแนน (เรียกจาก Pi)"""
    try:
        data = request.get_json()
        
        card_id = data.get('card_id')
        bottle_count = data.get('bottle_count', 0)
        can_count = data.get('can_count', 0)
        cap_count = data.get('cap_count', 0)
        label_count = data.get('label_count', 0)
        score = data.get('score', 0)
        image_path = data.get('image_path')
        
        if not card_id:
            return jsonify({'success': False, 'message': 'card_id required'}), 400
        
        success = add_or_update_card(
            card_id=card_id,
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
                'card_id': card_id,
                'score': score
            })
        else:
            return jsonify({'success': False, 'message': 'Database error'}), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cards')
def get_all_cards():
    """API ดึงข้อมูลบัตรทั้งหมด"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, card_name, total_score, scan_count, last_scan
        FROM rfid_cards 
        ORDER BY total_score DESC
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    card_list = []
    for card in cards:
        card_dict = {
            'card_id': card[0],
            'card_name': card[1] or f"Card-{card[0][:8]}",
            'total_score': card[2],
            'scan_count': card[3],
            'last_scan': card[4]
        }
        card_list.append(card_dict)
    
    return jsonify(card_list)

@app.route('/api/leaderboard')
def leaderboard():
    """API ตารางคะแนน"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, card_name, total_score, scan_count
        FROM rfid_cards 
        WHERE total_score > 0
        ORDER BY total_score DESC
        LIMIT 10
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    leaderboard = []
    for i, card in enumerate(cards, 1):
        card_dict = {
            'rank': i,
            'card_id': card[0],
            'card_name': card[1] or f"Card-{card[0][:8]}",
            'total_score': card[2],
            'scan_count': card[3]
        }
        leaderboard.append(card_dict)
    
    return jsonify(leaderboard)

if __name__ == '__main__':
    # สร้างโฟลเดอร์ templates
    os.makedirs('templates', exist_ok=True)
    
    # Initialize database
    init_database()
    
    print("🌐 เริ่ม Web Score System...")
    print("📊 URL: http://localhost:8000")
    print("📋 API Endpoints:")
    print("   - GET  /                    - หน้าแรก")
    print("   - GET  /card/<card_id>      - รายละเอียดบัตร")
    print("   - POST /api/add_score       - เพิ่มคะแนน")
    print("   - GET  /api/cards           - ข้อมูลบัตรทั้งหมด")
    print("   - GET  /api/leaderboard     - ตารางคะแนน")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
