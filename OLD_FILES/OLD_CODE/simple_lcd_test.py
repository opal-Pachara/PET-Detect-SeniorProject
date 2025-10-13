#!/usr/bin/env python3
"""
Simple LCD Test - ทดสอบ LCD แบบง่าย
"""

try:
    from RPLCD.i2c import CharLCD
    print("✅ RPLCD library found")
except ImportError:
    print("❌ RPLCD library not found")
    print("Install with: pip3 install RPLCD")
    exit(1)

import time

def test_lcd_basic():
    """ทดสอบ LCD แบบพื้นฐาน"""
    try:
        print("📺 Initializing LCD...")
        
        # สร้าง LCD object
        lcd = CharLCD('PCF8574', 0x27)  # Address 0x27
        print("✅ LCD initialized")
        
        # ทดสอบแสดงผล
        print("📝 Testing display...")
        
        # ล้างหน้าจอ
        lcd.clear()
        time.sleep(0.5)
        
        # แสดงข้อความบรรทัดแรก
        lcd.write_string("PET Detect System")
        time.sleep(1)
        
        # แสดงข้อความบรรทัดที่สอง
        lcd.cursor_pos = (1, 0)
        lcd.write_string("LCD Test OK!")
        time.sleep(2)
        
        # ทดสอบแสดงผลการตรวจจับ
        lcd.clear()
        lcd.write_string("Bottles: 2")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Score: 100")
        time.sleep(2)
        
        # ทดสอบแสดงข้อความยาว
        lcd.clear()
        lcd.write_string("Scanning...")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Place RFID card")
        time.sleep(2)
        
        # ล้างหน้าจอและปิด
        lcd.clear()
        lcd.write_string("Test Complete!")
        time.sleep(1)
        
        lcd.close()
        print("✅ LCD test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ LCD test failed: {e}")
        return False

def test_different_addresses():
    """ทดสอบ LCD addresses ที่เป็นไปได้"""
    addresses = [0x27, 0x3F, 0x20, 0x38]
    
    for addr in addresses:
        try:
            print(f"🔍 Testing address 0x{addr:02X}...")
            lcd = CharLCD('PCF8574', addr)
            lcd.clear()
            lcd.write_string(f"Address: 0x{addr:02X}")
            time.sleep(1)
            lcd.close()
            print(f"✅ LCD found at 0x{addr:02X}")
            return addr
        except Exception as e:
            print(f"❌ Address 0x{addr:02X} failed: {e}")
            continue
    
    print("❌ No LCD found at any address")
    return None

if __name__ == "__main__":
    print("=" * 50)
    print("📺 Simple LCD Test")
    print("=" * 50)
    
    # ทดสอบ address ต่างๆ
    working_addr = test_different_addresses()
    
    if working_addr:
        print(f"✅ LCD working at address 0x{working_addr:02X}")
        
        # ทดสอบการแสดงผล
        if test_lcd_basic():
            print("✅ LCD display working perfectly!")
        else:
            print("❌ LCD display test failed")
    else:
        print("❌ No LCD found. Check:")
        print("1. Wiring: VCC→5V, GND→GND, SDA→GPIO2, SCL→GPIO3")
        print("2. I2C enabled: sudo raspi-config")
        print("3. I2C tools: sudo apt install i2c-tools")
    
    print("🏁 Test complete!")
