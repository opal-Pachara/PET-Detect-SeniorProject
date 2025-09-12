#!/usr/bin/env python3
"""
Setup Database สำหรับ PET Detect Member System
รองรับทั้ง MySQL และ PostgreSQL
"""

import mysql.connector
from mysql.connector import Error
import os
import sys

# Database Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # เปลี่ยนเป็น MySQL user ของคุณ
    'password': '',  # ใส่ MySQL password ของคุณ
    'charset': 'utf8mb4'
}

DATABASE_NAME = 'pet_detect_db'
DB_USER = 'pet_user'
DB_PASSWORD = 'pet_password123'

def setup_mysql_database():
    """ตั้งค่า MySQL Database"""
    print("Setting up MySQL Database...")
    print("=" * 50)
    
    try:
        # 1. เชื่อมต่อ MySQL server
        print("1. Connecting to MySQL server...")
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # 2. สร้าง database
        print(f"2. Creating database: {DATABASE_NAME}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        # 3. สร้าง user และกำหนดสิทธิ์
        print(f"3. Creating user: {DB_USER}")
        try:
            cursor.execute(f"CREATE USER '{DB_USER}'@'localhost' IDENTIFIED BY '{DB_PASSWORD}'")
        except Error as e:
            if "already exists" in str(e):
                print(f"   User {DB_USER} already exists")
            else:
                raise e
        
        # 4. กำหนดสิทธิ์
        print("4. Granting privileges...")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {DATABASE_NAME}.* TO '{DB_USER}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # 5. ใช้ database
        cursor.execute(f"USE {DATABASE_NAME}")
        
        # 6. สร้างตาราง
        print("5. Creating tables...")
        
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
                INDEX idx_username (username),
                INDEX idx_score (total_score)
            )
        ''')
        print("   - members table created")
        
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
                INDEX idx_rfid (rfid_id),
                INDEX idx_timestamp (scan_timestamp)
            )
        ''')
        print("   - scan_logs table created")
        
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
        print("   - system_settings table created")
        
        # 7. ใส่ข้อมูลเริ่มต้น
        print("6. Inserting default settings...")
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
                INSERT IGNORE INTO system_settings (setting_key, setting_value, description) 
                VALUES (%s, %s, %s)
            ''', (key, value, desc))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("\nDatabase setup completed successfully!")
        print("=" * 50)
        print("Database Details:")
        print(f"   Host: {MYSQL_CONFIG['host']}")
        print(f"   Database: {DATABASE_NAME}")
        print(f"   User: {DB_USER}")
        print(f"   Password: {DB_PASSWORD}")
        print("\nConnection String:")
        print(f"   mysql://{DB_USER}:{DB_PASSWORD}@{MYSQL_CONFIG['host']}/{DATABASE_NAME}")
        
        return True
        
    except Error as e:
        print(f"MySQL setup error: {e}")
        return False

def test_database_connection():
    """ทดสอบการเชื่อมต่อ database"""
    print("\nTesting database connection...")
    
    try:
        config = {
            'host': MYSQL_CONFIG['host'],
            'database': DATABASE_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'charset': 'utf8mb4'
        }
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # ทดสอบ query
        cursor.execute("SELECT COUNT(*) FROM members")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scan_logs")
        log_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        print("Database connection test: SUCCESS")
        print(f"   Members: {count}")
        print(f"   Scan logs: {log_count}")
        
        return True
        
    except Error as e:
        print(f"Database connection test: FAILED")
        print(f"   Error: {e}")
        return False

def main():
    """Main function"""
    print("PET Detect Database Setup")
    print("=" * 50)
    print("This will setup MySQL database for member system")
    print("\nRequirements:")
    print("1. MySQL Server installed and running")
    print("2. MySQL root access (or admin user)")
    print("3. Python mysql-connector-python installed")
    print("   pip install mysql-connector-python")
    
    # เช็ค MySQL connector
    try:
        import mysql.connector
        print("\nMySQL connector: OK")
    except ImportError:
        print("\nMySQL connector: NOT FOUND")
        print("Please install: pip install mysql-connector-python")
        return
    
    # ขอข้อมูล MySQL
    print(f"\nCurrent MySQL config:")
    print(f"   Host: {MYSQL_CONFIG['host']}")
    print(f"   User: {MYSQL_CONFIG['user']}")
    print(f"   Password: {'*' * len(MYSQL_CONFIG['password']) if MYSQL_CONFIG['password'] else '(empty)'}")
    
    proceed = input("\nProceed with database setup? (y/N): ").lower().strip()
    if proceed not in ['y', 'yes']:
        print("Setup cancelled")
        return
    
    # Setup database
    if setup_mysql_database():
        # Test connection
        test_database_connection()
        
        print("\nNext steps:")
        print("1. Update database config in member_system.py")
        print("2. Run: python member_system.py")
        print("3. Open: http://localhost:9000")
    else:
        print("Database setup failed!")

if __name__ == "__main__":
    main()
