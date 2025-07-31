#!/bin/bash

echo "🔧 ติดตั้งระบบ RFID + Camera + API บน Raspberry Pi"
echo "=================================================="

# Update system
echo "📦 อัปเดตระบบ..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 ติดตั้ง system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    cmake \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtcore4 \
    libqtgui4 \
    libqt4-test \
    v4l-utils \
    libv4l-dev

# Enable SPI for RFID
echo "🔌 เปิดใช้งาน SPI..."
sudo raspi-config nonint do_spi 0

# Install USB Camera tools
echo "📷 ติดตั้ง USB Camera tools..."
sudo apt-get install -y v4l-utils

# Enable I2C (if needed)
echo "🔌 เปิดใช้งาน I2C..."
sudo raspi-config nonint do_i2c 0

# Create virtual environment
echo "🐍 สร้าง Python virtual environment..."
python3 -m venv pet_detect_env
source pet_detect_env/bin/activate

# Install Python dependencies
echo "📦 ติดตั้ง Python dependencies..."
pip install --upgrade pip
pip install -r requirements_pi.txt

# Create directories
echo "📁 สร้างโฟลเดอร์..."
mkdir -p images
mkdir -p logs

# Set permissions
echo "🔐 ตั้งค่าสิทธิ์..."
chmod +x rfid_camera_system.py

echo "✅ ติดตั้งเสร็จสิ้น!"
echo ""
echo "🚀 วิธีใช้งาน:"
echo "1. เปิด terminal"
echo "2. cd /path/to/project"
echo "3. source pet_detect_env/bin/activate"
echo "4. python rfid_camera_system.py"
echo ""
echo "📋 การเชื่อมต่อ Hardware:"
echo "- RFID RC522: ตามตารางที่ให้มา"
echo "- LED: GPIO 18"
echo "- Buzzer: GPIO 12"
echo "- Camera: USB Camera"
echo ""
echo "🔍 ทดสอบ USB Camera:"
echo "python test_usb_camera.py" 