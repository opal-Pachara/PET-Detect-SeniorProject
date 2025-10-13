#!/usr/bin/env python3
"""
ตรวจสอบสถานะ GPIO และรีเซ็ต LED
"""

import RPi.GPIO as GPIO
import time

# GPIO Pin สำหรับ LED
LED_PIN = 2

def check_and_reset_gpio():
    """ตรวจสอบและรีเซ็ต GPIO"""
    try:
        # ตั้งค่า GPIO mode
        GPIO.setmode(GPIO.BCM)
        
        # ตรวจสอบสถานะปัจจุบัน
        print(f"ตรวจสอบ GPIO {LED_PIN}...")
        
        # ตั้งค่าเป็น OUTPUT
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        # อ่านสถานะปัจจุบัน
        current_state = GPIO.input(LED_PIN)
        print(f"สถานะปัจจุบัน GPIO {LED_PIN}: {'HIGH (ON)' if current_state else 'LOW (OFF)'}")
        
        # บังคับปิด LED
        print("บังคับปิด LED...")
        GPIO.output(LED_PIN, GPIO.LOW)
        
        # ตรวจสอบอีกครั้ง
        new_state = GPIO.input(LED_PIN)
        print(f"สถานะใหม่ GPIO {LED_PIN}: {'HIGH (ON)' if new_state else 'LOW (OFF)'}")
        
        # ทดสอบเปิด-ปิด
        print("\nทดสอบควบคุม LED:")
        for i in range(3):
            print(f"รอบ {i+1}: เปิด LED")
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(1)
            print(f"รอบ {i+1}: ปิด LED")
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(1)
        
        print("ทดสอบเสร็จ - LED ควรปิดแล้ว")
        
    except Exception as e:
        print(f"ข้อผิดพลาด: {e}")
    finally:
        try:
            GPIO.output(LED_PIN, GPIO.LOW)  # บังคับปิด
            GPIO.cleanup()
            print("GPIO cleanup เสร็จสิ้น")
        except:
            pass

if __name__ == "__main__":
    print("ตรวจสอบและรีเซ็ต GPIO สำหรับ LED")
    print("=" * 40)
    check_and_reset_gpio()
