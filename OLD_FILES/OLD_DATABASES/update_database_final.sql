-- อัพเดท Database สำหรับ Member System
-- รันใน phpMyAdmin: http://localhost/phpmyadmin

USE pet_detect_db;

-- 1. เพิ่มคอลัมน์ password_hash ถ้ายังไม่มี
ALTER TABLE members ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255) AFTER username;

-- 2. ตรวจสอบโครงสร้างตาราง
DESCRIBE members;

-- 3. ตรวจสอบข้อมูลปัจจุบัน
SELECT rfid_id, username, password_hash, full_name, email, phone FROM members LIMIT 10;

-- 4. อัพเดทข้อมูลสมาชิกที่มีอยู่แล้วให้มี password_hash เป็น NULL
UPDATE members SET password_hash = NULL WHERE password_hash = '';

-- 5. ตรวจสอบผลลัพธ์
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
FROM members;
