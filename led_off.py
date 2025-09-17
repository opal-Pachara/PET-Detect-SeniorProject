#!/usr/bin/env python3
"""
สคริปต์ปิด LED ที่ GPIO 2
ใช้เมื่อ LED ติดค้างหรือต้องการปิดก่อนเริ่มระบบ
"""

import RPi.GPIO as GPIO
import time

LED_PIN = 2  # GPIO 2

def turn_off_led():
    """ปิด LED"""
    try:
        print("กำลังปิด LED...")
        
        # ตั้งค่า GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        # บังคับปิด LED (Active Low - ใช้ HIGH เพื่อปิด)
        GPIO.output(LED_PIN, GPIO.HIGH)
        print(f"LED ที่ GPIO {LED_PIN} ปิดแล้ว (Active Low)")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # ทำความสะอาด GPIO
        try:
            GPIO.cleanup()
            print("GPIO cleanup complete")
        except:
            pass

if __name__ == "__main__":
    print("LED OFF Script")
    print("-" * 20)
    turn_off_led()
