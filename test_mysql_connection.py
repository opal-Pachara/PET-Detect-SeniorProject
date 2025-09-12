#!/usr/bin/env python3
"""
ทดสอบการเชื่อมต่อ MySQL Database
"""

import mysql.connector
from mysql.connector import Error

def test_mysql_connection():
    """ทดสอบการเชื่อมต่อ MySQL"""
    print("Testing MySQL Connection...")
    print("=" * 40)
    
    # Database Configuration
    config = {
        'host': 'localhost',
        'database': 'pet_detect_db',
        'user': 'pet_user',
        'password': 'pet_password123',
        'charset': 'utf8mb4'
    }
    
    try:
        # เชื่อมต่อฐานข้อมูล
        print("1. Connecting to MySQL...")
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            print("   ✅ MySQL connection successful!")
            
            # ตรวจสอบ database
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE()")
            database = cursor.fetchone()[0]
            print(f"   📊 Database: {database}")
            
            # ตรวจสอบตาราง
            print("\n2. Checking tables...")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            expected_tables = ['members', 'scan_logs', 'system_settings']
            for table in expected_tables:
                if (table,) in tables:
                    print(f"   ✅ Table '{table}' exists")
                else:
                    print(f"   ❌ Table '{table}' missing")
            
            # ตรวจสอบข้อมูล
            print("\n3. Checking data...")
            cursor.execute("SELECT COUNT(*) FROM members")
            member_count = cursor.fetchone()[0]
            print(f"   👥 Members: {member_count}")
            
            cursor.execute("SELECT COUNT(*) FROM scan_logs")
            log_count = cursor.fetchone()[0]
            print(f"   📝 Scan logs: {log_count}")
            
            cursor.execute("SELECT COUNT(*) FROM system_settings")
            setting_count = cursor.fetchone()[0]
            print(f"   ⚙️ Settings: {setting_count}")
            
            # ทดสอบ insert
            print("\n4. Testing insert...")
            test_rfid = "TEST123456789"
            cursor.execute("""
                INSERT IGNORE INTO members (rfid_id, username, full_name) 
                VALUES (%s, %s, %s)
            """, (test_rfid, f"user_{test_rfid[:8]}", "Test User"))
            
            if cursor.rowcount > 0:
                print("   ✅ Insert test successful")
            else:
                print("   ℹ️ Test user already exists")
            
            # ลบ test data
            cursor.execute("DELETE FROM members WHERE rfid_id = %s", (test_rfid,))
            print("   🧹 Test data cleaned up")
            
            cursor.close()
            connection.close()
            
            print("\n🎉 MySQL Database is ready!")
            return True
            
    except Error as e:
        print(f"❌ MySQL connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if XAMPP MySQL is running")
        print("2. Check if database 'pet_detect_db' exists")
        print("3. Check if user 'pet_user' exists")
        print("4. Run create_database.sql in phpMyAdmin")
        return False

def main():
    """Main function"""
    print("PET Detect MySQL Connection Test")
    print("=" * 40)
    
    if test_mysql_connection():
        print("\n✅ Ready to run member system!")
        print("Run: python member_system.py")
    else:
        print("\n❌ Please fix MySQL issues first")
        print("1. Open phpMyAdmin: http://localhost/phpmyadmin")
        print("2. Run create_database.sql")
        print("3. Test again: python test_mysql_connection.py")

if __name__ == "__main__":
    main()
