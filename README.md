# PET Detect Senior Project

ระบบตรวจจับขวด PET และการจัดการคะแนนด้วย RFID

## 🎯 ฟีเจอร์หลัก

### 🤖 AI Detection
- ตรวจจับขวด PET, กระป๋อง, ฝา, สลาก
- ใช้ YOLOv5 model
- คำนวณคะแนนอัตโนมัติ

### 🏷️ RFID System
- สแกน RFID card
- สร้างสมาชิกอัตโนมัติ
- จัดการคะแนนส่วนตัว

### 🎮 Hardware Control
- ควบคุม Stepper Motor
- เรียงขวดตามประเภท
- ระบบกล้อง USB

### 🌐 Web Interface
- ระบบสมาชิก
- ตารางคะแนน
- ประวัติการสแกน

## 🚀 การติดตั้ง

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ติดตั้ง MySQL
- ดาวน์โหลด XAMPP: https://www.apachefriends.org/download.html
- เปิด Apache และ MySQL
- รัน `create_database.sql` ใน phpMyAdmin

### 3. รันระบบ
```bash
# AI API
python minimal_api.py

# Member System
python member_system.py

# Pi Client
python pi_client_subprocess.py
```

## 📁 โครงสร้างไฟล์

```
PET-Detect-SeniorProject/
├── code/
│   ├── api.py                 # AI API (Flask)
│   ├── main.py               # Main app
│   └── ...
├── model-yolov11/
│   └── best.pt               # YOLOv5 model
├── templates/
│   ├── login.html            # หน้า Login
│   ├── register.html         # หน้าสมัครสมาชิก
│   └── members.html          # ตารางคะแนน
├── pi_client_subprocess.py   # Pi Client
├── member_system.py          # Member System
├── minimal_api.py            # AI API
├── create_database.sql       # Database setup
└── requirements.txt          # Dependencies
```

## 🔧 การใช้งาน

### 1. เข้าสู่ระบบ
- เปิด: http://localhost:9000
- ใส่ RFID ID
- ตั้งรหัสผ่าน (ครั้งแรก)

### 2. สแกน RFID
- Pi Client จะสแกน RFID อัตโนมัติ
- สร้างสมาชิกใหม่ (ถ้ายังไม่มี)
- บันทึกคะแนนลงฐานข้อมูล

### 3. ดูคะแนน
- หน้า Dashboard: http://localhost:9000/dashboard
- รายละเอียดสมาชิก: http://localhost:9000/member/<rfid_id>

## 🗄️ Database Schema

### Members Table
```sql
CREATE TABLE members (
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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### Scan Logs Table
```sql
CREATE TABLE scan_logs (
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
    FOREIGN KEY (member_id) REFERENCES members(id) ON DELETE CASCADE
);
```

## 🔌 API Endpoints

### AI API (Port 5000)
- `POST /scan_image` - วิเคราะห์ภาพ
- `GET /ping` - ตรวจสอบสถานะ

### Member System (Port 9000)
- `GET /` - หน้า Login
- `GET /dashboard` - ตารางคะแนน
- `GET /register` - หน้าสมัครสมาชิก
- `POST /api/add_score` - เพิ่มคะแนน
- `POST /api/check_member` - ตรวจสอบสมาชิก

## 🎮 Hardware Setup

### Raspberry Pi
- RFID Reader (MFRC522)
- USB Camera
- Stepper Motor + Driver
- GPIO Pins: 18, 19 (Stepper), 10, 11, 12, 13 (RFID)

### Wiring
```
Stepper Motor:
- STEP → GPIO 18
- DIR → GPIO 19
- ENA → GPIO (optional)

RFID:
- SDA → GPIO 10
- SCK → GPIO 11
- MOSI → GPIO 12
- MISO → GPIO 13
- IRQ → GPIO (not used)
- GND → GND
- RST → GPIO 15
- 3.3V → 3.3V
```

## 📊 คะแนน

- ขวด PET: +50 คะแนน
- กระป๋อง: +100 คะแนน
- ฝา: -10 คะแนน
- สลาก: -10 คะแนน

## 🐛 Troubleshooting

### MySQL Connection Error
1. ตรวจสอบ XAMPP MySQL ทำงาน
2. รัน `create_database.sql` ใน phpMyAdmin
3. ตรวจสอบ user `pet_user` และ password

### Pi Client Error
1. ตรวจสอบ GPIO pins
2. ตรวจสอบ USB camera
3. ตรวจสอบ network connection

### AI Model Error
1. ตรวจสอบไฟล์ `model-yolov11/best.pt`
2. ตรวจสอบ dependencies: `torch`, `ultralytics`

## 👥 Contributors

- Senior Project Team
- PET Detection System

## 📄 License

MIT License

## 📞 Support

หากมีปัญหาการใช้งาน กรุณาติดต่อทีมพัฒนา