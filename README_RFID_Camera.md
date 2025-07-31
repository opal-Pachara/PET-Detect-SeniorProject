# 🎯 PET Detection System - RFID + Camera + API

ระบบตรวจจับขวด PET ที่รวม RFID reader, camera และ cloud API

## 📋 ระบบการทำงาน

```
1. 🔍 สแกน RFID Card
2. 📸 ถ่ายภาพด้วย Camera
3. 📡 ส่งภาพไปยัง Cloud API
4. 📊 รับผลการวิเคราะห์กลับมา
5. 💡 แสดงผลด้วย LED
```

## 🛠️ Hardware Requirements

### Raspberry Pi
- Raspberry Pi 4 (แนะนำ) หรือ Pi 3B+
- MicroSD Card 16GB+
- Power Supply 5V/3A

### Sensors & Components
- **RFID Reader:** MFRC522 (SPI)
- **Camera:** 
  - PiCamera (CSI port) หรือ
  - USB Camera
- **LEDs:** 
  - Green LED (GPIO 18) - Success
  - Red LED (GPIO 17) - Error  
  - Blue LED (GPIO 27) - Scanning

## 🔌 Hardware Connections

### RFID Reader (MFRC522)
```
VCC   → 3.3V
GND   → GND
SDA   → GPIO 8 (CE0)
SCK   → GPIO 11 (SCLK)
MOSI  → GPIO 10 (MOSI)
MISO  → GPIO 9 (MISO)
RST   → GPIO 25
```

### LEDs
```
Green LED  → GPIO 18 (ผ่าน Resistor 220Ω)
Red LED    → GPIO 17 (ผ่าน Resistor 220Ω)
Blue LED   → GPIO 27 (ผ่าน Resistor 220Ω)
```

### Camera
- **PiCamera:** เชื่อมต่อ CSI port
- **USB Camera:** เชื่อมต่อ USB port

## 📦 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd PET-Detect-SeniorProject
```

### 2. Run Setup Script
```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

### 3. Install Python Dependencies
```bash
pip3 install -r requirements_pi.txt
```

## 🚀 Usage

### สำหรับ PiCamera
```bash
python3 rfid_camera_system.py
```

### สำหรับ USB Camera
```bash
python3 rfid_camera_system_usb.py
```

## 📊 System Flow

### 1. RFID Scanning
- ระบบรอการสแกน RFID card
- Blue LED เปิดแสดงสถานะ scanning
- เมื่อสแกนสำเร็จจะได้ Card ID และ Card Text

### 2. Image Capture
- หลังจากสแกน RFID สำเร็จ
- ระบบจะถ่ายภาพด้วย camera
- บันทึกไฟล์ใน `/home/pi/images/`

### 3. API Communication
- ส่งภาพไปยัง cloud API
- พร้อมข้อมูล RFID (Card ID, Card Text)
- รับผลการวิเคราะห์กลับมา

### 4. Result Display
- แสดงผลการนับ (ขวด, ฝา, สลาก)
- แสดงคะแนน
- LED แสดงสถานะ:
  - 🟢 Green: สำเร็จ
  - 🔴 Red: ผิดพลาด
  - 🔵 Blue: กำลังทำงาน

## 📁 File Structure

```
PET-Detect-SeniorProject/
├── rfid_camera_system.py          # PiCamera version
├── rfid_camera_system_usb.py      # USB Camera version
├── requirements_pi.txt             # Pi dependencies
├── setup_pi.sh                    # Setup script
├── README_RFID_Camera.md          # This file
└── /home/pi/images/               # Captured images
```

## 🔧 Configuration

### API URL
แก้ไขในไฟล์ `rfid_camera_system.py`:
```python
api_url="https://your-api-url.com/api/scan"
```

### Camera Index (USB Camera)
แก้ไขในไฟล์ `rfid_camera_system_usb.py`:
```python
camera_index=0  # หรือ 1, 2 ตามลำดับ
```

### LED GPIO Pins
แก้ไขในไฟล์:
```python
self.led_green = 18  # Green LED
self.led_red = 17     # Red LED  
self.led_blue = 27    # Blue LED
```

## 🐛 Troubleshooting

### RFID ไม่ทำงาน
1. ตรวจสอบการเชื่อมต่อ SPI
2. รัน `sudo raspi-config` → Interface Options → SPI → Enable
3. ตรวจสอบการเชื่อมต่อสายไฟ

### Camera ไม่ทำงาน
1. **PiCamera:** ตรวจสอบ CSI port
2. **USB Camera:** ตรวจสอบ USB port
3. รัน `lsusb` เพื่อดู USB devices
4. รัน `v4l2-ctl --list-devices` เพื่อดู camera devices

### API ไม่เชื่อมต่อ
1. ตรวจสอบ internet connection
2. ตรวจสอบ API URL
3. ทดสอบด้วย `curl` หรือ Postman

### LED ไม่ทำงาน
1. ตรวจสอบการเชื่อมต่อ GPIO
2. ตรวจสอบ resistor (220Ω)
3. ทดสอบด้วย `gpio` command

## 📈 API Response Format

```json
{
  "bottle_count": 2,
  "cap_count": 1, 
  "label_count": 0,
  "score": 70,
  "message": "Detection completed successfully"
}
```

## 🔄 Loop Process

```
┌─────────────────┐
│   Start System  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Scan RFID Card │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Capture Image   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Send to API     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Display Result  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Wait 5 seconds  │
└─────────┬───────┘
          │
          ▼
    ┌─────────┐
    │  Loop   │
    └─────────┘
```

## 🎯 Features

- ✅ **RFID Card Scanning**
- ✅ **Image Capture** (PiCamera/USB)
- ✅ **Cloud API Integration**
- ✅ **LED Status Indicators**
- ✅ **Error Handling**
- ✅ **Automatic Loop**
- ✅ **Timestamp Logging**

## 📞 Support

หากมีปัญหาหรือต้องการความช่วยเหลือ:
1. ตรวจสอบ log ใน terminal
2. ตรวจสอบ hardware connections
3. ทดสอบแต่ละส่วนแยกกัน

---

**�� Happy Coding! 🎉** 