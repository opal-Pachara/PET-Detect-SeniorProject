#!/usr/bin/env python3
"""
โค้ดปิดไฟ LED แยก (GPIO 4)
ใช้สำหรับปิดไฟ LED ที่ต่อกับ GPIO 4
"""

import RPi.GPIO as GPIO
import time


LED_PIN = 4  # GPIO 4

def setup_gpio():
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        print(f" GPIO {LED_PIN} setup complete (OFF)")
    except Exception as e:
        print(f" GPIO setup error: {e}")

def turn_off_led():
    
    try:
        GPIO.output(LED_PIN, GPIO.LOW)
        print(f" LED OFF - GPIO {LED_PIN}")
    except Exception as e:
        print(f" LED OFF error: {e}")

def cleanup():
    
    try:
        GPIO.cleanup()
        print(" GPIO cleanup complete")
    except Exception as e:
        print(f" GPIO cleanup error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print(" LED OFF Script (GPIO 4)")
    print("=" * 50)
    
    try:
        
        setup_gpio()
        
        
        turn_off_led()
        
        
        time.sleep(1)
        
        print(" LED turned OFF successfully!")
        
    except KeyboardInterrupt:
        print("\n Script interrupted by user")
    except Exception as e:
        print(f" Script error: {e}")
    finally:
        
        cleanup()
        print(" Script finished")