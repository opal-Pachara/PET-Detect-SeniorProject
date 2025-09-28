#!/usr/bin/env python3
"""
Setup I2C LCD Display
"""

import smbus
import time

# I2C address ของ LCD (ปกติเป็น 0x27 หรือ 0x3F)
LCD_ADDRESS = 0x27

def scan_i2c():
    """สแกนหา I2C devices"""
    try:
        bus = smbus.SMBus(1)  # I2C bus 1
        print("🔍 Scanning I2C devices...")
        
        devices = []
        for addr in range(0x03, 0x78):
            try:
                bus.read_byte(addr)
                devices.append(hex(addr))
                print(f"✅ Found device at {hex(addr)}")
            except:
                pass
        
        if devices:
            print(f"📱 Found {len(devices)} I2C devices: {devices}")
        else:
            print("❌ No I2C devices found")
            
        return devices
        
    except Exception as e:
        print(f"❌ I2C scan error: {e}")
        return []

def test_lcd():
    """ทดสอบ LCD Display"""
    try:
        from RPLCD.i2c import CharLCD
        
        # สร้าง LCD object
        lcd = CharLCD('PCF8574', LCD_ADDRESS)
        
        # ทดสอบแสดงข้อความ
        lcd.clear()
        lcd.write_string("Hello World!")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("LCD Test OK!")
        
        print("✅ LCD Display working!")
        time.sleep(3)
        
        # ล้างหน้าจอ
        lcd.clear()
        lcd.close()
        
    except ImportError:
        print("❌ RPLCD library not installed")
        print("Install with: pip3 install RPLCD")
    except Exception as e:
        print(f"❌ LCD test error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("📺 LCD I2C Setup Script")
    print("=" * 50)
    
    # สแกนหา I2C devices
    devices = scan_i2c()
    
    if devices:
        # ทดสอบ LCD
        test_lcd()
    else:
        print("❌ No I2C devices found. Check wiring!")
    
    print("🏁 Setup complete!")
