#!/usr/bin/env python3
"""
Simple Web Score System - ไม่ใช้ debug mode
"""

from flask import Flask, jsonify, request
import sqlite3
import json
from datetime import datetime
import os

app = Flask(__name__)

# Database file
DB_FILE = 'rfid_scores.db'

def init_database():
    """สร้างฐานข้อมูล"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rfid_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT UNIQUE NOT NULL,
            total_score INTEGER DEFAULT 0,
            scan_count INTEGER DEFAULT 0,
            last_scan TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT NOT NULL,
            bottle_count INTEGER DEFAULT 0,
            can_count INTEGER DEFAULT 0,
            cap_count INTEGER DEFAULT 0,
            label_count INTEGER DEFAULT 0,
            score INTEGER DEFAULT 0,
            scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

@app.route('/')
def index():
    """หน้าแรก - JSON data"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, total_score, scan_count, last_scan
        FROM rfid_cards 
        ORDER BY total_score DESC
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    card_list = []
    for card in cards:
        card_dict = {
            'card_id': card[0],
            'total_score': card[1],
            'scan_count': card[2],
            'last_scan': card[3]
        }
        card_list.append(card_dict)
    
    return jsonify({
        'success': True,
        'total_cards': len(cards),
        'cards': card_list
    })

@app.route('/api/add_score', methods=['POST'])
def add_score():
    """API สำหรับเพิ่มคะแนน"""
    try:
        data = request.get_json()
        
        card_id = data.get('card_id')
        bottle_count = data.get('bottle_count', 0)
        can_count = data.get('can_count', 0)
        cap_count = data.get('cap_count', 0)
        label_count = data.get('label_count', 0)
        score = data.get('score', 0)
        
        if not card_id:
            return jsonify({'success': False, 'message': 'card_id required'}), 400
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
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
        else:
            # เพิ่มบัตรใหม่
            cursor.execute('''
                INSERT INTO rfid_cards (card_id, total_score, scan_count, last_scan)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ''', (card_id, score))
        
        # เพิ่มประวัติการสแกน
        cursor.execute('''
            INSERT INTO scan_history (card_id, bottle_count, can_count, cap_count, label_count, score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (card_id, bottle_count, can_count, cap_count, label_count, score))
        
        conn.commit()
        conn.close()
        
        print(f"💾 Score saved: Card {card_id}, Score +{score}")
        
        return jsonify({
            'success': True,
            'message': 'Score added successfully',
            'card_id': card_id,
            'score': score
        })
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cards')
def get_all_cards():
    """ดูข้อมูลบัตรทั้งหมด"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, total_score, scan_count, last_scan
        FROM rfid_cards 
        ORDER BY total_score DESC
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    card_list = []
    for card in cards:
        card_dict = {
            'card_id': card[0],
            'total_score': card[1],
            'scan_count': card[2],
            'last_scan': card[3]
        }
        card_list.append(card_dict)
    
    return jsonify(card_list)

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    print("🌐 Simple Web Score System...")
    print("📊 URL: http://localhost:8000")
    print("📋 API Endpoints:")
    print("   - GET  /                    - ข้อมูลบัตรทั้งหมด")
    print("   - POST /api/add_score       - เพิ่มคะแนน")
    print("   - GET  /api/cards           - ข้อมูลบัตรทั้งหมด")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
