#!/usr/bin/env python3
"""
Setup MySQL Database สำหรับ PET Detect Member System
"""

import mysql.connector
from mysql.connector import Error
import os
import sys

def setup_mysql_database():
    """ตั้งค่า MySQL Database"""
    print("Setting up MySQL Database...")
    print("=" * 50)
    
    # ขอข้อมูล MySQL
    print("MySQL Configuration:")
    host = input("Host (localhost): ").strip() or "localhost"
    user = input("MySQL User (root): ").strip() or "root"
    password = input("MySQL Password: ").strip()
    
    database_name = "pet_detect_db"
    db_user = "pet_user"
    db_password = "pet_password123"
    
    try:
        # 1. เชื่อมต่อ MySQL server
        print(f"\n1. Connecting to MySQL server ({host})...")
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # 2. สร้าง database
        print(f"2. Creating database: {database_name}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        # 3. สร้าง user และกำหนดสิทธิ์
        print(f"3. Creating user: {db_user}")
        try:
            cursor.execute(f"CREATE USER '{db_user}'@'localhost' IDENTIFIED BY '{db_password}'")
        except Error as e:
            if "already exists" in str(e):
                print(f"   User {db_user} already exists")
            else:
                raise e
        
        # 4. กำหนดสิทธิ์
        print("4. Granting privileges...")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {database_name}.* TO '{db_user}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # 5. ใช้ database
        cursor.execute(f"USE {database_name}")
        
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
        print(f"   Host: {host}")
        print(f"   Database: {database_name}")
        print(f"   User: {db_user}")
        print(f"   Password: {db_password}")
        print("\nConnection String:")
        print(f"   mysql://{db_user}:{db_password}@{host}/{database_name}")
        
        # อัพเดท config file
        update_config_file(host, db_user, db_password, database_name)
        
        return True
        
    except Error as e:
        print(f"MySQL setup error: {e}")
        return False

def update_config_file(host, user, password, database):
    """อัพเดทไฟล์ config"""
    config_content = f'''# Database Configuration
DB_CONFIG = {{
    'host': '{host}',
    'database': '{database}',
    'user': '{user}',
    'password': '{password}',
    'charset': 'utf8mb4'
}}'''
    
    # อัพเดท member_system.py
    try:
        with open('member_system.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แทนที่ DB_CONFIG
        import re
        pattern = r'# Database Configuration\nDB_CONFIG = \{.*?\}'
        new_content = re.sub(pattern, config_content, content, flags=re.DOTALL)
        
        with open('member_system.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"\nConfig updated in member_system.py")
        
    except Exception as e:
        print(f"Config update error: {e}")

def test_connection():
    """ทดสอบการเชื่อมต่อ"""
    print("\nTesting database connection...")
    
    try:
        # อ่าน config จากไฟล์
        with open('member_system.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract DB_CONFIG
        import re
        pattern = r"DB_CONFIG = \{([^}]+)\}"
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("Cannot find DB_CONFIG in member_system.py")
            return False
        
        config_str = match.group(1)
        config = {}
        for line in config_str.split('\n'):
            if ':' in line and "'" in line:
                key = line.split(':')[0].strip().strip("'")
                value = line.split(':')[1].strip().strip("',")
                config[key] = value
        
        # ทดสอบเชื่อมต่อ
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
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
        
    except Exception as e:
        print(f"Database connection test: FAILED")
        print(f"   Error: {e}")
        return False

def main():
    """Main function"""
    print("PET Detect MySQL Database Setup")
    print("=" * 50)
    print("This will setup MySQL database for member system")
    print("\nRequirements:")
    print("1. MySQL Server installed and running")
    print("2. MySQL root access (or admin user)")
    print("3. Python mysql-connector-python installed")
    
    # เช็ค MySQL connector
    try:
        import mysql.connector
        print("\nMySQL connector: OK")
    except ImportError:
        print("\nMySQL connector: NOT FOUND")
        print("Please install: pip install mysql-connector-python")
        return
    
    proceed = input("\nProceed with database setup? (y/N): ").lower().strip()
    if proceed not in ['y', 'yes']:
        print("Setup cancelled")
        return
    
    # Setup database
    if setup_mysql_database():
        # Test connection
        test_connection()
        
        print("\nNext steps:")
        print("1. Run: python member_system.py")
        print("2. Open: http://localhost:9000")
        print("3. Test with Pi client")
    else:
        print("Database setup failed!")

if __name__ == "__main__":
    main()
