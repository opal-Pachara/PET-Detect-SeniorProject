#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

def check_database():
    """ตรวจสอบ database และข้อมูล"""
    try:
        # เชื่อมต่อ database
        connection = mysql.connector.connect(
            host='localhost',
            database='pet_detect_db',
            user='pet_user',
            password='pet_password123'
        )
        
        if connection.is_connected():
            print("✅ เชื่อมต่อ MySQL สำเร็จ")
            
            cursor = connection.cursor()
            
            # ตรวจสอบโครงสร้างตาราง
            print("\n📋 โครงสร้างตาราง members:")
            cursor.execute("DESCRIBE members")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[0]} - {col[1]}")
            
            # ตรวจสอบข้อมูลสมาชิก
            print("\n👥 ข้อมูลสมาชิก:")
            cursor.execute("SELECT rfid_id, username, password_hash, full_name, email, phone FROM members")
            members = cursor.fetchall()
            
            if members:
                for member in members:
                    print(f"  RFID: {member[0]}")
                    print(f"  Username: {member[1]}")
                    print(f"  Password Hash: {member[2][:20] if member[2] else 'None'}...")
                    print(f"  Full Name: {member[3]}")
                    print(f"  Email: {member[4]}")
                    print(f"  Phone: {member[5]}")
                    print("  " + "-"*50)
            else:
                print("  ไม่มีข้อมูลสมาชิก")
            
            cursor.close()
            connection.close()
            print("\n✅ ตรวจสอบเสร็จสิ้น")
            
    except Error as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    check_database()
