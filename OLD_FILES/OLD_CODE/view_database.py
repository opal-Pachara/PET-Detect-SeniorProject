#!/usr/bin/env python3
"""
ดูข้อมูลใน Database
"""

import sqlite3
from datetime import datetime

def view_all_cards():
    """ดูข้อมูลบัตรทั้งหมด"""
    conn = sqlite3.connect('rfid_scores.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, total_score, scan_count, last_scan
        FROM rfid_cards 
        ORDER BY total_score DESC
    ''')
    
    cards = cursor.fetchall()
    conn.close()
    
    print("📊 ข้อมูลบัตร RFID ทั้งหมด:")
    print("=" * 60)
    print(f"{'Card ID':<15} {'คะแนนรวม':<10} {'จำนวนสแกน':<12} {'สแกนล่าสุด':<20}")
    print("-" * 60)
    
    for card in cards:
        card_id = card[0][:12] + "..." if len(card[0]) > 12 else card[0]
        print(f"{card_id:<15} {card[1]:<10} {card[2]:<12} {card[3][:19] if card[3] else 'ไม่มี':<20}")
    
    print(f"\nรวม: {len(cards)} บัตร")

def view_recent_scans(limit=10):
    """ดูการสแกนล่าสุด"""
    conn = sqlite3.connect('rfid_scores.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, bottle_count, can_count, score, scan_timestamp
        FROM scan_history 
        ORDER BY scan_timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    scans = cursor.fetchall()
    conn.close()
    
    print(f"\n📋 การสแกนล่าสุด {limit} ครั้ง:")
    print("=" * 70)
    print(f"{'Card ID':<15} {'ขวด':<6} {'กระป๋อง':<8} {'คะแนน':<8} {'วันเวลา':<20}")
    print("-" * 70)
    
    for scan in scans:
        card_id = scan[0][:12] + "..." if len(scan[0]) > 12 else scan[0]
        print(f"{card_id:<15} {scan[1]:<6} {scan[2]:<8} {scan[3]:<8} {scan[4][:19]:<20}")

def view_top_cards(limit=5):
    """ดูบัตรคะแนนสูงสุด"""
    conn = sqlite3.connect('rfid_scores.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT card_id, total_score, scan_count
        FROM rfid_cards 
        WHERE total_score > 0
        ORDER BY total_score DESC
        LIMIT ?
    ''', (limit,))
    
    cards = cursor.fetchall()
    conn.close()
    
    print(f"\n🏆 Top {limit} บัตรคะแนนสูงสุด:")
    print("=" * 50)
    
    for i, card in enumerate(cards, 1):
        emoji = ["🥇", "🥈", "🥉", "🏅", "🎖️"][min(i-1, 4)]
        card_id = card[0][:12] + "..." if len(card[0]) > 12 else card[0]
        print(f"{emoji} {i}. {card_id} - {card[1]} คะแนน ({card[2]} ครั้ง)")

def main():
    """Main function"""
    print("🔍 ดูข้อมูล RFID Score Database")
    print("=" * 60)
    
    try:
        view_all_cards()
        view_recent_scans(10)
        view_top_cards(5)
        
        print(f"\n💾 Database file: rfid_scores.db")
        print(f"🌐 Web interface: http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Database ยังไม่ได้สร้าง ให้รัน web_score_system.py ก่อน")

if __name__ == "__main__":
    main()
