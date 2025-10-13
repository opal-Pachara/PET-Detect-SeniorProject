#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

def update_database():
    """อัพเดท Database สำหรับ Member System"""
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
            
            # 1. เพิ่มคอลัมน์ password_hash ถ้ายังไม่มี
            try:
                cursor.execute("ALTER TABLE members ADD COLUMN password_hash VARCHAR(255) AFTER username")
                print("✅ เพิ่มคอลัมน์ password_hash สำเร็จ")
            except Error as e:
                if "Duplicate column name" in str(e):
                    print("ℹ️ คอลัมน์ password_hash มีอยู่แล้ว")
                else:
                    print(f"❌ เกิดข้อผิดพลาด: {e}")
            
            # 2. ตรวจสอบโครงสร้างตาราง
            print("\n📋 โครงสร้างตาราง members:")
            cursor.execute("DESCRIBE members")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[0]} - {col[1]}")
            
            # 3. อัพเดทข้อมูลสมาชิกที่มีอยู่แล้ว
            cursor.execute("UPDATE members SET password_hash = NULL WHERE password_hash = ''")
            updated_rows = cursor.rowcount
            print(f"\n🔄 อัพเดทข้อมูลสมาชิก {updated_rows} รายการ")
            
            # 4. ตรวจสอบข้อมูลสมาชิก
            print("\n👥 ข้อมูลสมาชิก:")
            cursor.execute("""
                SELECT 
                    rfid_id, 
                    username, 
                    CASE 
                        WHEN password_hash IS NULL OR password_hash = '' THEN 'No Password'
                        ELSE 'Has Password'
                    END as password_status,
                    full_name, 
                    email, 
                    phone 
                FROM members
            """)
            members = cursor.fetchall()
            
            if members:
                for member in members:
                    print(f"  RFID: {member[0]}")
                    print(f"  Username: {member[1]}")
                    print(f"  Password Status: {member[2]}")
                    print(f"  Full Name: {member[3]}")
                    print(f"  Email: {member[4]}")
                    print(f"  Phone: {member[5]}")
                    print("  " + "-"*50)
            else:
                print("  ไม่มีข้อมูลสมาชิก")
            
            # Commit changes
            connection.commit()
            cursor.close()
            connection.close()
            print("\n✅ อัพเดท Database เสร็จสิ้น")
            
    except Error as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    update_database()
