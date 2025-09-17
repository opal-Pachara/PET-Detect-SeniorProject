#!/usr/bin/env python3
"""
ทดสอบ GPIO 2 ว่าสามารถควบคุม LED ได้หรือไม่
"""

import RPi.GPIO as GPIO
import time
import sys

LED_PIN = 2  # GPIO 2

def test_gpio():
    """ทดสอบ GPIO"""
    try:
        print("เริ่มทดสอบ GPIO 2...")
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        print("ทดสอบ 1: ส่งสัญญาณ LOW (ควรเปิด LED)")
        GPIO.output(LED_PIN, GPIO.LOW)
        input("กด Enter เมื่อตรวจสอบ LED แล้ว...")
        
        print("ทดสอบ 2: ส่งสัญญาณ HIGH (ควรปิด LED)")
        GPIO.output(LED_PIN, GPIO.HIGH)
        input("กด Enter เมื่อตรวจสอบ LED แล้ว...")
        
        print("ทดสอบ 3: กระพริบ 5 ครั้ง")
        for i in range(5):
            print(f"รอบที่ {i+1}: LOW (เปิด)")
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(1)
            print(f"รอบที่ {i+1}: HIGH (ปิด)")
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(1)
        
        print("ทดสอบเสร็จสิ้น!")
        
    except KeyboardInterrupt:
        print("\nหยุดการทดสอบ")
    except Exception as e:
        print(f"ข้อผิดพลาด: {e}")
    finally:
        GPIO.output(LED_PIN, GPIO.HIGH)  # ปิด LED
        GPIO.cleanup()
        print("GPIO cleanup complete")

if __name__ == "__main__":
    print("GPIO 2 LED Test")
    print("=" * 30)
    print("ตรวจสอบว่า LED เปลี่ยนแปลงตามสัญญาณ GPIO หรือไม่")
    print("หาก LED ไม่เปลี่ยน = ปัญหาการต่อสาย")
    print("หาก LED เปลี่ยนแต่กลับกัน = ต้องเปลี่ยน Active High/Low")
    print("-" * 30)
    
    test_gpio()
