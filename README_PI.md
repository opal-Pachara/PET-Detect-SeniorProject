# 🎯 ระบบ RFID + Camera + API สำหรับ Raspberry Pi

ระบบสแกน RFID แล้วถ่ายภาพส่งไปยัง Cloud API เพื่อวิเคราะห์ขวด PET

## 📋 ฟีเจอร์

- ✅ **สแกน RFID** - อ่านบัตร RFID ก่อน
- ✅ **ถ่ายภาพ** - ใช้ PiCamera หลังจากสแกน RFID
- ✅ **ส่ง API** - ส่งภาพและข้อมูล RFID ไปยัง Cloud
- ✅ **แสดงผล** - LED และ Buzzer สำหรับ feedback
- ✅ **Loop อัตโนมัติ** - ทำงานต่อเนื่อง

## 🔧 การติดตั้ง

### 1. Clone โปรเจค
```bash
git clone <your-repo>
cd PET-Detect-SeniorProject
```

### 2. รัน Setup Script
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

### 3. เปิดใช้งาน Virtual Environment
```bash
source pet_detect_env/bin/activate
```

## 🔌 การเชื่อมต่อ Hardware

### RFID RC522 Module
| RC522 Pin | Breadboard | Raspberry Pi GPIO | Raspberry Pi Pin |
|-----------|------------|-------------------|------------------|
| 3.3V      | VCC row    | 3.3V              | Pin 1            |
| GND       | GND row    | GND               | Pin 6            |
| SDA       | SDA row    | GPIO 8 (CE0)      | Pin 24           |
| SCK       | SCK row    | GPIO 11 (SCLK)    | Pin 23           |
| MOSI      | MOSI row   | GPIO 10 (MOSI)    | Pin 19           |
| MISO      | MISO row   | GPIO 9 (MISO)     | Pin 21           |
| RST       | RST row    | GPIO 25           | Pin 22           |

### LED และ Buzzer
- **LED:** GPIO 18 (แสดงสถานะ)
- **Buzzer:** GPIO 12 (เสียงแจ้งเตือน)

### Camera
- **USB Camera:** เชื่อมต่อผ่าน USB port
- รองรับ camera index 0 และ 1

## 🚀 การใช้งาน

### รันระบบ
```bash
python rfid_camera_system.py
```

### กระบวนการทำงาน
1. **วางบัตร RFID** บนเครื่องอ่าน
2. **ระบบสแกน** และอ่าน Card ID
3. **ถ่ายภาพ** วัตถุที่ต้องการวิเคราะห์
4. **ส่งข้อมูล** ไปยัง Cloud API
5. **แสดงผล** การวิเคราะห์ (ขวด, ฝา, สลาก, คะแนน)
6. **รอรอบถัดไป** (5 วินาที)

### การหยุดทำงาน
กด `Ctrl+C` เพื่อหยุดการทำงาน

## 📊 ผลลัพธ์

### API Response
```json
{
  "bottle_count": 2,
  "cap_count": 1,
  "label_count": 0,
  "score": 70,
  "message": "Detection completed successfully"
}
```

### Visual Feedback
- **LED กระพริบ 3 ครั้ง:** สำเร็จ
- **LED กระพริบ 5 ครั้ง:** เกิดข้อผิดพลาด
- **Buzzer:** เสียงแจ้งเตือนเมื่อเสร็จสิ้น

## 🔧 การแก้ไขปัญหา

### RFID ไม่ทำงาน
1. ตรวจสอบการเชื่อมต่อ SPI
2. รัน `sudo raspi-config` → Interface Options → SPI → Enable
3. ตรวจสอบการเชื่อมต่อสายไฟ

### Camera ไม่ทำงาน
1. ตรวจสอบการเชื่อมต่อ USB Camera
2. รัน `lsusb` เพื่อดู USB devices
3. ตรวจสอบ camera index (0 หรือ 1)
4. รัน `v4l2-ctl --list-devices` เพื่อดู camera devices

### API ไม่เชื่อมต่อ
1. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
2. ตรวจสอบ URL ในโค้ด
3. ตรวจสอบ Railway deployment

## 📁 โครงสร้างไฟล์

```
PET-Detect-SeniorProject/
├── rfid_camera_system.py    # ระบบหลัก
├── requirements_pi.txt      # Python dependencies
├── setup_pi.sh             # Setup script
├── README_PI.md            # คู่มือนี้
├── images/                 # ภาพที่ถ่าย
└── logs/                   # ไฟล์ log
```

## 🛠️ การปรับแต่ง

### เปลี่ยน GPIO Pins
แก้ไขใน `rfid_camera_system.py`:
```python
LED_PIN = 18      # เปลี่ยนเป็น GPIO ที่ต้องการ
BUZZER_PIN = 12   # เปลี่ยนเป็น GPIO ที่ต้องการ
```

### เปลี่ยน API URL
แก้ไขใน `rfid_camera_system.py`:
```python
API_URL = "https://your-api-url.com/api/scan"
```

### เปลี่ยน Camera Settings
แก้ไขใน `__init__` method:
```python
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ความกว้าง
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # ความสูง
self.camera.set(cv2.CAP_PROP_FPS, 30)             # FPS
```

## 📞 การสนับสนุน

หากมีปัญหาหรือต้องการความช่วยเหลือ:
1. ตรวจสอบ log files
2. ตรวจสอบการเชื่อมต่อ hardware
3. ตรวจสอบ Railway deployment status

---

**🎯 ระบบพร้อมใช้งาน! วางบัตร RFID และเริ่มต้นการทำงานได้เลย** 