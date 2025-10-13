-- สร้าง Database สำหรับ PET Detect Member System
-- รันใน phpMyAdmin หรือ MySQL Command Line

-- 1. สร้าง Database
CREATE DATABASE IF NOT EXISTS pet_detect_db 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 2. ใช้ Database
USE pet_detect_db;

-- 3. สร้าง User (ถ้ายังไม่มี)
CREATE USER IF NOT EXISTS 'pet_user'@'localhost' IDENTIFIED BY 'pet_password123';
GRANT ALL PRIVILEGES ON pet_detect_db.* TO 'pet_user'@'localhost';
FLUSH PRIVILEGES;

-- 4. สร้างตารางสมาชิก
CREATE TABLE IF NOT EXISTS members (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rfid_id VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255),
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
);

-- 5. สร้างตารางประวัติการสแกน
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
);

-- 6. สร้างตารางการตั้งค่า
CREATE TABLE IF NOT EXISTS system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 7. ใส่ข้อมูลเริ่มต้น
INSERT IGNORE INTO system_settings (setting_key, setting_value, description) VALUES
('bottle_score', '50', 'คะแนนสำหรับขวด'),
('can_score', '100', 'คะแนนสำหรับกระป๋อง'),
('cap_penalty', '-10', 'หักคะแนนสำหรับฝา'),
('label_penalty', '-10', 'หักคะแนนสำหรับสลาก'),
('system_name', 'PET Detect Score System', 'ชื่อระบบ'),
('auto_register', 'true', 'สมัครสมาชิกอัตโนมัติเมื่อสแกน RFID ใหม่');

-- 8. ตรวจสอบตาราง
SHOW TABLES;
SELECT COUNT(*) as member_count FROM members;
SELECT COUNT(*) as log_count FROM scan_logs;
SELECT COUNT(*) as setting_count FROM system_settings;
