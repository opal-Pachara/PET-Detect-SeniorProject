-- อัพเดท Database สำหรับ Member System
-- รันใน phpMyAdmin

USE pet_detect_db;

-- เพิ่มคอลัมน์ password_hash ถ้ายังไม่มี
ALTER TABLE members ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255) AFTER username;

-- ตรวจสอบตาราง
DESCRIBE members;

-- ตรวจสอบข้อมูล
SELECT rfid_id, username, password_hash, full_name, email, phone FROM members LIMIT 5;
