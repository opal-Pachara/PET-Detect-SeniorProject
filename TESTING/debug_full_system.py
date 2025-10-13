#!/usr/bin/env python3
"""
Debug ระบบเต็ม - ทีละขั้นตอน
"""

import time
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rfid():
    """ทดสอบ RFID"""
    print("1️⃣ ทดสอบ RFID...")
    try:
        from mfrc522 import SimpleMFRC522
        reader = SimpleMFRC522()
        print("✅ RFID OK")
        return True
    except Exception as e:
        print(f"❌ RFID Error: {e}")
        return False

def test_stepper():
    """ทดสอบ Stepper Motor"""
    print("\n2️⃣ ทดสอบ Stepper Motor...")
    try:
        from stepper_motor_controller import StepperMotorController
        stepper = StepperMotorController(
            step_pin=18,
            dir_pin=19,
            enable_pin=None
        )
        print("✅ Stepper OK")
        stepper.cleanup()
        return True
    except Exception as e:
        print(f"❌ Stepper Error: {e}")
        return False

def test_camera():
    """ทดสอบ Camera"""
    print("\n3️⃣ ทดสอบ Camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera OK")
                return True
            else:
                print("❌ Camera: Can't capture frame")
                return False
        else:
            print("❌ Camera: Can't open")
            return False
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return False

def test_api():
    """ทดสอบ API Connection"""
    print("\n4️⃣ ทดสอบ API...")
    try:
        import requests
        response = requests.get("http://192.168.1.31:5000/api/ping", timeout=5)
        if response.status_code == 200:
            print("✅ API OK")
            return True
        else:
            print(f"❌ API: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def main():
    """Main debug function"""
    print("🔧 Debug ระบบเต็ม")
    print("=" * 50)
    
    tests = [
        ("RFID", test_rfid),
        ("Stepper", test_stepper), 
        ("Camera", test_camera),
        ("API", test_api)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} Test Failed: {e}")
            results[name] = False
    
    print("\n📊 สรุปผลการทดสอบ:")
    print("=" * 30)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:10}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 ผลรวม: {'✅ ทุกอย่างพร้อม' if all_passed else '❌ มีปัญหาบางส่วน'}")
    
    if all_passed:
        print("\n🚀 ลองรันระบบเต็ม:")
        print("python pi_client_with_stepper.py")
    else:
        print("\n🔧 แก้ปัญหาที่ FAIL ก่อน")

if __name__ == "__main__":
    main()