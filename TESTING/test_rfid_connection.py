#!/usr/bin/env python3
"""
ทดสอบการเชื่อมต่อ RFID MFRC522
"""

import RPi.GPIO as GPIO
import time
import sys

def test_rfid_basic():
    """ทดสอบ RFID connection พื้นฐาน"""
    print("🔍 ทดสอบการเชื่อมต่อ RFID MFRC522")
    print("=" * 50)
    
    try:
        from mfrc522 import SimpleMFRC522
        print("✅ Import SimpleMFRC522 สำเร็จ")
        
        # สร้าง RFID reader
        reader = SimpleMFRC522()
        print("✅ สร้าง RFID reader สำเร็จ")
        
        print("\n📋 การต่อสาย RFID:")
        print("SDA  → GPIO 8  (Pin 24)")
        print("SCK  → GPIO 11 (Pin 23)")
        print("MOSI → GPIO 10 (Pin 19)")
        print("MISO → GPIO 9  (Pin 21)")
        print("RST  → GPIO 25 (Pin 22)")
        print("GND  → GND     (Pin 6)")
        print("3.3V → 3.3V    (Pin 1)")
        
        print(f"\n🔄 รอสแกน RFID card/tag...")
        print("กดปุ่ม Ctrl+C เพื่อหยุด")
        
        timeout = 10  # timeout 10 วินาที
        start_time = time.time()
        
        while True:
            try:
                # ตรวจสอบ timeout
                if time.time() - start_time > timeout:
                    print(f"\n⏰ Timeout {timeout} วินาที - ไม่พบ RFID card")
                    print("\n🔧 แนะนำการแก้ปัญหา:")
                    print("1. ตรวจสอบการต่อสาย")
                    print("2. เปิด SPI ใน raspi-config")
                    print("3. ทดสอบด้วย RFID card อื่น")
                    break
                
                # อ่าน RFID
                print(".", end="", flush=True)
                id, text = reader.read_no_block()
                
                if id is not None:
                    print(f"\n🎉 พบ RFID card!")
                    print(f"Card ID: {id}")
                    print(f"Text: '{text.strip()}'")
                    print("✅ RFID ทำงานปกติ!")
                    break
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print(f"\n🛑 หยุดการทดสอบ")
                break
            except Exception as e:
                print(f"\n❌ เกิดข้อผิดพลาด: {e}")
                break
                
    except ImportError:
        print("❌ ไม่พบ mfrc522 library")
        print("ติดตั้งด้วย: pip install mfrc522")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        
    finally:
        try:
            GPIO.cleanup()
        except:
            pass

def test_spi_interface():
    """ทดสอบ SPI interface"""
    print("\n🔧 ทดสอบ SPI interface:")
    
    import os
    
    # เช็ค SPI devices
    spi_devices = []
    if os.path.exists("/dev/spidev0.0"):
        spi_devices.append("/dev/spidev0.0")
    if os.path.exists("/dev/spidev0.1"):
        spi_devices.append("/dev/spidev0.1")
    
    if spi_devices:
        print(f"✅ พบ SPI devices: {spi_devices}")
    else:
        print("❌ ไม่พบ SPI devices")
        print("เปิด SPI ด้วย: sudo raspi-config > Interfacing Options > SPI > Enable")
    
    # เช็ค SPI modules
    try:
        result = os.popen("lsmod | grep spi").read()
        if "spi" in result:
            print(f"✅ SPI modules loaded")
        else:
            print("❌ SPI modules ไม่ได้โหลด")
    except:
        pass

if __name__ == "__main__":
    test_spi_interface()
    test_rfid_basic()