#!/usr/bin/env python3
"""
ทดสอบการเชื่อมต่อมอเตอร์กับ Raspberry Pi
ใช้สำหรับตรวจสอบก่อนรันระบบจริง
"""

import RPi.GPIO as GPIO
import time

# กำหนด GPIO pins (ปรับให้ตรงกับการต่อสายจริง)
MOTOR_PIN1 = 16  # IN1 หรือ A-IA
MOTOR_PIN2 = 18  # IN2 หรือ A-IB
ENABLE_PIN = 22  # ENA (สำหรับ L298N เท่านั้น, L9110S ไม่ต้องใช้)

def setup_gpio():
    """ตั้งค่า GPIO"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN1, GPIO.OUT)
    GPIO.setup(MOTOR_PIN2, GPIO.OUT)
    
    # สำหรับ L298N
    if ENABLE_PIN:
        GPIO.setup(ENABLE_PIN, GPIO.OUT)
        pwm = GPIO.PWM(ENABLE_PIN, 100)
        pwm.start(0)
        return pwm
    return None

def test_motor_directions(pwm=None):
    """ทดสอบการหมุนมอเตอร์"""
    
    print("🔧 เริ่มทดสอบมอเตอร์...")
    
    try:
        # เปิด PWM (สำหรับ L298N)
        if pwm:
            pwm.ChangeDutyCycle(70)  # ความเร็ว 70%
        
        # ทดสอบหมุนซ้าย
        print("⬅️  ทดสอบหมุนซ้าย (3 วินาที)...")
        GPIO.output(MOTOR_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR_PIN2, GPIO.LOW)
        time.sleep(3)
        
        # หยุด
        print("⏸️  หยุด (1 วินาที)...")
        GPIO.output(MOTOR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_PIN2, GPIO.LOW)
        time.sleep(1)
        
        # ทดสอบหมุนขวา
        print("➡️  ทดสอบหมุนขวา (3 วินาที)...")
        GPIO.output(MOTOR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_PIN2, GPIO.HIGH)
        time.sleep(3)
        
        # หยุด
        print("⏸️  หยุดมอเตอร์")
        GPIO.output(MOTOR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_PIN2, GPIO.LOW)
        
        if pwm:
            pwm.ChangeDutyCycle(0)
        
        print("✅ การทดสอบเสร็จสิ้น!")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    
def test_motor_speeds(pwm=None):
    """ทดสอบความเร็วต่างๆ (สำหรับ L298N เท่านั้น)"""
    
    if not pwm:
        print("⚠️  PWM ไม่พร้อม - ข้ามการทดสอบความเร็ว")
        return
    
    print("\n🚀 ทดสอบความเร็วต่างๆ...")
    
    speeds = [30, 50, 70, 100]
    
    for speed in speeds:
        print(f"⚡ ความเร็ว {speed}% (2 วินาที)...")
        
        pwm.ChangeDutyCycle(speed)
        GPIO.output(MOTOR_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR_PIN2, GPIO.LOW)
        time.sleep(2)
        
        # หยุด
        GPIO.output(MOTOR_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_PIN2, GPIO.LOW)
        pwm.ChangeDutyCycle(0)
        time.sleep(0.5)
    
    print("✅ การทดสอบความเร็วเสร็จสิ้น!")

def main():
    print("🔌 ทดสอบการเชื่อมต่อมอเตอร์กับ Raspberry Pi")
    print("=" * 50)
    print(f"📍 GPIO Pins:")
    print(f"   Motor Pin 1: GPIO {MOTOR_PIN1}")
    print(f"   Motor Pin 2: GPIO {MOTOR_PIN2}")
    print(f"   Enable Pin:  GPIO {ENABLE_PIN if ENABLE_PIN else 'ไม่ใช้'}")
    print("=" * 50)
    
    try:
        # ตั้งค่า GPIO
        pwm = setup_gpio()
        
        # ทดสอบการหมุน
        test_motor_directions(pwm)
        
        # ทดสอบความเร็ว (ถ้ามี PWM)
        test_motor_speeds(pwm)
        
    except KeyboardInterrupt:
        print("\n🛑 การทดสอบถูกยกเลิก")
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    
    finally:
        # ปิด GPIO
        GPIO.cleanup()
        print("🔌 GPIO cleaned up")

if __name__ == "__main__":
    main()