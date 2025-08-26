#!/usr/bin/env python3
"""
ทดสอบ Professional Stepper Motor Driver
สำหรับ Driver ที่ใช้ DIP switches ตั้งค่า micro step
"""

from stepper_motor_controller import StepperMotorController
import time
import RPi.GPIO as GPIO

def test_professional_driver():
    print("🔧 ทดสอบ Professional Stepper Motor Driver")
    print("=" * 60)
    
    try:
        # สร้าง controller (ไม่ใช้ ENA pin)
        stepper = StepperMotorController(
            step_pin=20,    # PUL+ → GPIO 20
            dir_pin=21,     # DIR+ → GPIO 21
            enable_pin=None # ENA+ → ไม่ต้องต่อ
        )
        
        # ให้ผู้ใช้ใส่ค่า steps ตาม DIP switch
        print("\n📋 กรุณาตั้งค่า DIP switches บน driver ก่อน:")
        print("   - 3200 steps = 1/16 micro step (แนะนำ)")
        print("   - 1600 steps = 1/8 micro step")
        print("   - 800 steps = 1/4 micro step")
        
        while True:
            try:
                steps_input = input("\nใส่จำนวน steps/revolution ตาม DIP switch (เช่น 3200): ")
                steps_per_rev = int(steps_input)
                if steps_per_rev > 0:
                    stepper.set_steps_per_revolution(steps_per_rev)
                    break
                else:
                    print("กรุณาใส่ตัวเลขที่มากกว่า 0")
            except ValueError:
                print("กรุณาใส่ตัวเลขที่ถูกต้อง")
        
        print(f"\n✅ ใช้การตั้งค่า: {steps_per_rev} steps/revolution")
        print(f"📐 ความละเอียด: {360/steps_per_rev:.4f} องศา/step")
        
        # ทดสอบการหมุน
        print("\n🔄 เริ่มทดสอบการหมุน...")
        input("กด Enter เพื่อเริ่ม...")
        
        # ทดสอบ 1: หมุนขวา 90 องศา
        print("1️⃣ หมุนขวา 90 องศา...")
        stepper.rotate_right(90, speed=1000)
        time.sleep(2)
        
        # ทดสอบ 2: หมุนซ้าย 180 องศา  
        print("2️⃣ หมุนซ้าย 180 องศา...")
        stepper.rotate_left(180, speed=1000)
        time.sleep(2)
        
        # ทดสอบ 3: หมุนขวา 45 องศา (ความแม่นยำ)
        print("3️⃣ หมุนขวา 45 องศา (ทดสอบความแม่นยำ)...")
        stepper.rotate_right(45, speed=800)
        time.sleep(2)
        
        # ทดสอบ 4: กลับตำแหน่งเริ่มต้น
        print("4️⃣ กลับตำแหน่งเริ่มต้น...")
        stepper.return_to_home(speed=1200)
        time.sleep(1)
        
        print(f"\n📐 ตำแหน่งสุดท้าย: {stepper.get_position_degrees()}°")
        print("✅ การทดสอบเสร็จสิ้น!")
        
        # ทดสอบความเร็วต่างๆ
        print("\n🚀 ทดสอบความเร็วต่างๆ...")
        speeds = [500, 1000, 1500, 2000]
        
        for speed in speeds:
            print(f"⚡ ความเร็ว {speed} steps/sec...")
            stepper.move_degrees(30, speed)
            time.sleep(0.5)
            stepper.move_degrees(-30, speed)
            time.sleep(0.5)
        
        print("✅ การทดสอบความเร็วเสร็จสิ้น!")
        
    except KeyboardInterrupt:
        print("\n🛑 การทดสอบถูกยกเลิก")
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    
    finally:
        try:
            stepper.cleanup()
            GPIO.cleanup()
            print("🔌 GPIO cleaned up")
        except:
            pass

if __name__ == "__main__":
    test_professional_driver()