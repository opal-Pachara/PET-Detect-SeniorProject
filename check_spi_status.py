#!/usr/bin/env python3
"""
เช็คสถานะ SPI Interface บน Raspberry Pi
"""

import os
import subprocess

def check_spi_status():
    """เช็คสถานะ SPI อย่างละเอียด"""
    print("🔍 ตรวจสอบสถานะ SPI Interface")
    print("=" * 50)
    
    # 1. เช็ค SPI devices
    print("\n1️⃣ SPI Devices:")
    spi_devices = []
    
    if os.path.exists("/dev/spidev0.0"):
        spi_devices.append("/dev/spidev0.0")
        print("✅ /dev/spidev0.0 - พบ")
    else:
        print("❌ /dev/spidev0.0 - ไม่พบ")
    
    if os.path.exists("/dev/spidev0.1"):
        spi_devices.append("/dev/spidev0.1")
        print("✅ /dev/spidev0.1 - พบ")
    else:
        print("❌ /dev/spidev0.1 - ไม่พบ")
    
    # 2. เช็ค SPI modules ใน kernel
    print("\n2️⃣ SPI Kernel Modules:")
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if result.returncode == 0:
            spi_modules = []
            for line in result.stdout.split('\n'):
                if 'spi' in line.lower():
                    spi_modules.append(line.strip())
            
            if spi_modules:
                print("✅ SPI modules loaded:")
                for module in spi_modules:
                    print(f"   {module}")
            else:
                print("❌ ไม่พบ SPI modules")
        else:
            print("❌ ไม่สามารถเช็ค kernel modules ได้")
    except Exception as e:
        print(f"❌ Error checking modules: {e}")
    
    # 3. เช็ค config.txt
    print("\n3️⃣ Config.txt Settings:")
    config_file = "/boot/config.txt"
    
    # เช็คที่อาจจะอยู่
    possible_configs = ["/boot/config.txt", "/boot/firmware/config.txt"]
    config_found = False
    
    for config_path in possible_configs:
        if os.path.exists(config_path):
            config_file = config_path
            config_found = True
            break
    
    if config_found:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                
            print(f"📁 Config file: {config_file}")
            
            # เช็ค SPI settings
            if 'dtparam=spi=on' in content:
                print("✅ dtparam=spi=on - พบ")
            elif 'dtparam=spi=off' in content:
                print("❌ dtparam=spi=off - SPI ปิดอยู่")
            else:
                print("⚠️ dtparam=spi - ไม่ระบุ (ใช้ default)")
            
            # เช็ค SPI overlays
            spi_overlays = []
            for line in content.split('\n'):
                if 'spi' in line.lower() and ('dtoverlay' in line or 'dtparam' in line):
                    spi_overlays.append(line.strip())
            
            if spi_overlays:
                print("\n📋 SPI related settings:")
                for overlay in spi_overlays:
                    print(f"   {overlay}")
            
        except Exception as e:
            print(f"❌ ไม่สามารถอ่าน config.txt: {e}")
    else:
        print("❌ ไม่พบ config.txt")
    
    # 4. เช็ค GPIO permissions
    print("\n4️⃣ GPIO Permissions:")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        print("✅ RPi.GPIO accessible")
        GPIO.cleanup()
    except Exception as e:
        print(f"❌ RPi.GPIO error: {e}")
    
    # 5. เช็ค SPI Python libraries
    print("\n5️⃣ Python SPI Libraries:")
    
    # spidev
    try:
        import spidev
        print("✅ spidev library - พร้อมใช้งาน")
    except ImportError:
        print("❌ spidev library - ไม่พบ")
        print("   ติดตั้ง: pip install spidev")
    
    # mfrc522
    try:
        from mfrc522 import SimpleMFRC522
        print("✅ mfrc522 library - พร้อมใช้งาน")
    except ImportError:
        print("❌ mfrc522 library - ไม่พบ")
        print("   ติดตั้ง: pip install mfrc522")
    
    # 6. สรุปผล
    print("\n📊 สรุปผล:")
    if spi_devices:
        if len(spi_devices) >= 1:
            print("✅ SPI Interface: เปิดใช้งานแล้ว")
            print(f"   Devices: {', '.join(spi_devices)}")
        else:
            print("⚠️ SPI Interface: เปิดบางส่วน")
    else:
        print("❌ SPI Interface: ปิดอยู่")
        print("\n🔧 วิธีแก้ไข:")
        print("1. sudo raspi-config")
        print("2. เลือก 'Interfacing Options' หรือ 'Interface Options'")
        print("3. เลือก 'SPI'")
        print("4. เลือก 'Yes' เพื่อเปิดใช้งาน")
        print("5. Reboot: sudo reboot")

def test_spi_communication():
    """ทดสอบการสื่อสาร SPI แบบง่าย"""
    print("\n🧪 ทดสอบการสื่อสาร SPI:")
    
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)  # bus 0, device 0
        spi.max_speed_hz = 1000000
        spi.mode = 0
        
        # ส่งข้อมูลทดสอบ
        resp = spi.xfer2([0x00])
        print(f"✅ SPI communication test: response = {resp}")
        
        spi.close()
    except Exception as e:
        print(f"❌ SPI communication test failed: {e}")

if __name__ == "__main__":
    check_spi_status()
    test_spi_communication()