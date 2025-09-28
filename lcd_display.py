#!/usr/bin/env python3
"""
LCD Display สำหรับแสดงผล PET Detection
"""

import time
from RPLCD.i2c import CharLCD

class LCDDisplay:
    def __init__(self, address=0x27):
        """สร้าง LCD Display object"""
        try:
            self.lcd = CharLCD('PCF8574', address)
            self.lcd.clear()
            self.lcd.write_string("PET Detect System")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string("Ready...")
            print("✅ LCD Display initialized")
        except Exception as e:
            print(f"❌ LCD init error: {e}")
            self.lcd = None
    
    def show_scanning(self):
        """แสดงสถานะกำลังสแกน"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.write_string("Scanning...")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string("Place RFID card")
    
    def show_results(self, bottle_count, cap_count, score):
        """แสดงผลการตรวจจับ"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.write_string(f"Bottles: {bottle_count}")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(f"Score: {score}")
    
    def show_error(self, message):
        """แสดงข้อผิดพลาด"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.write_string("Error:")
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(message[:16])  # จำกัด 16 ตัวอักษร
    
    def clear(self):
        """ล้างหน้าจอ"""
        if self.lcd:
            self.lcd.clear()
    
    def close(self):
        """ปิด LCD"""
        if self.lcd:
            self.lcd.clear()
            self.lcd.close()

# ทดสอบ LCD
if __name__ == "__main__":
    print("📺 Testing LCD Display...")
    
    try:
        lcd = LCDDisplay()
        
        # ทดสอบแสดงผล
        lcd.show_scanning()
        time.sleep(2)
        
        lcd.show_results(2, 1, 110)
        time.sleep(3)
        
        lcd.clear()
        lcd.write_string("Test Complete!")
        time.sleep(2)
        
        lcd.close()
        print("✅ LCD test complete!")
        
    except Exception as e:
        print(f"❌ LCD test error: {e}")
