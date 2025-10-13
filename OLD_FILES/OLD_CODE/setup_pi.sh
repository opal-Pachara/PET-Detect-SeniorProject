#!/bin/bash

# Setup script สำหรับ Raspberry Pi Client
# ติดตั้ง dependencies และตั้งค่าระบบ

echo "🥧 PET Detect - Raspberry Pi Setup"
echo "=================================="

# ตรวจสอบว่าเป็น Raspberry Pi หรือไม่
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "   Some GPIO functions may not work properly"
fi

# อัพเดทระบบ
echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ติดตั้ง system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils \
    libv4l-dev

# เปิดใช้งาน SPI สำหรับ RFID
echo "🔧 Enabling SPI interface..."
if ! grep -q "dtparam=spi=on" /boot/config.txt; then
    echo "dtparam=spi=on" | sudo tee -a /boot/config.txt
    echo "✅ SPI enabled (reboot required)"
else
    echo "✅ SPI already enabled"
fi

# สร้าง virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# เปิดใช้งาน virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# อัพเกรด pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# ติดตั้ง Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements_pi.txt" ]; then
    pip install -r requirements_pi.txt
    echo "✅ Python dependencies installed"
else
    echo "❌ requirements_pi.txt not found"
    echo "   Installing basic dependencies..."
    pip install requests opencv-python mfrc522 RPi.GPIO
fi

# ตรวจสอบ camera
echo "📷 Checking camera devices..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✅ Camera devices found:"
    ls /dev/video*
    
    # ทดสอบ camera
    echo "🔍 Testing camera with v4l2..."
    v4l2-ctl --list-devices
else
    echo "❌ No camera devices found"
    echo "   Please connect a USB camera"
fi

# สร้างโฟลเดอร์ที่จำเป็น
echo "📁 Creating necessary directories..."
mkdir -p captured_images
mkdir -p logs

# ตั้งค่าสิทธิ์ไฟล์
echo "🔑 Setting file permissions..."
chmod +x pi_client.py

# แสดงข้อมูลการใช้งาน
echo ""
echo "🎉 Raspberry Pi setup completed!"
echo "================================"
echo "📋 Next steps:"
echo "   1. Connect RFID reader to SPI pins:"
echo "      - SDA  -> Pin 24 (GPIO 8)"
echo "      - SCK  -> Pin 23 (GPIO 11)"
echo "      - MOSI -> Pin 19 (GPIO 10)"
echo "      - MISO -> Pin 21 (GPIO 9)"
echo "      - IRQ  -> Not connected"
echo "      - GND  -> Pin 6 (Ground)"
echo "      - RST  -> Pin 22 (GPIO 25)"
echo "      - 3.3V -> Pin 1 (3.3V Power)"
echo ""
echo "   2. Connect USB camera"
echo ""
echo "   3. Update API URL in pi_client.py:"
echo "      API_URL = 'http://YOUR_SERVER_IP:5000'"
echo ""
echo "   4. Run the client:"
echo "      source venv/bin/activate"
echo "      python pi_client.py"
echo ""
echo "⚠️  Note: Reboot required if SPI was just enabled"
echo "   sudo reboot"