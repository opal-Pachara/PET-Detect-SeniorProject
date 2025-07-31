#!/usr/bin/env python3
"""
Simple RFID + Camera + API System
ใช้ code เดิมเป็นฐาน แล้วเพิ่ม RFID
"""

import cv2
import requests
import time
from mfrc522 import SimpleMFRC522

def scan_rfid():
    """สแกน RFID และ return card ID"""
    try:
        print("🔍 กำลังสแกน RFID...")
        rfid_reader = SimpleMFRC522()
        card_id, text = rfid_reader.read()
        print(f"✅ สแกนสำเร็จ! Card ID: {card_id}")
        return card_id
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการสแกน RFID: {e}")
        return None

def capture_and_send():
    """ถ่ายภาพและส่งไป API (ใช้ code เดิม)"""
    # Open the USB camera (usually index 0 for the first camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ USB camera not found.")
        return False

    # Capture a frame from the camera
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image.")
        return False

    # Save the captured image to a file
    image_path = "usb_image.jpg"
    cv2.imwrite(image_path, frame)
    print(f"✅ Image saved: {image_path}")

    # Send the image to the API
    url = "https://pet-detect-seniorproject-production.up.railway.app/api/scan"
    with open(image_path, "rb") as img_file:
        files = {'image': img_file}
        response = requests.post(url, files=files)

    print(f"📡 Status code: {response.status_code}")
    try:
        result = response.json()
        print("📊 Response:")
        print(f"   - ขวด: {result.get('result', {}).get('bottle_count', 0)}")
        print(f"   - ฝา: {result.get('result', {}).get('cap_count', 0)}")
        print(f"   - สลาก: {result.get('result', {}).get('label_count', 0)}")
        print(f"   - คะแนน: {result.get('result', {}).get('score', 0)}")
        return True
    except Exception as e:
        print(f"❌ Response error: {response.text}")
        return False

def main():
    """Main function - สแกน RFID แล้วถ่ายภาพ"""
    print("🎯 Simple RFID + Camera + API System")
    print("=" * 50)
    
    while True:
        try:
            print("\n" + "="*50)
            print("📋 วางบัตร RFID บนเครื่องอ่าน...")
            
            # Step 1: Scan RFID
            card_id = scan_rfid()
            
            if card_id is None:
                print("❌ ไม่สามารถสแกน RFID ได้ ลองใหม่อีกครั้ง")
                time.sleep(2)
                continue
            
            # Step 2: Capture and send (ใช้ code เดิม)
            print("\n📸 กำลังถ่ายภาพและส่งไปยัง API...")
            success = capture_and_send()
            
            if success:
                print("✅ กระบวนการเสร็จสิ้น!")
            else:
                print("❌ เกิดข้อผิดพลาดในการส่งข้อมูล")
            
            # Wait before next cycle
            print("\n⏳ รอ 5 วินาทีก่อนรอบถัดไป...")
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n🛑 หยุดการทำงาน...")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main() 