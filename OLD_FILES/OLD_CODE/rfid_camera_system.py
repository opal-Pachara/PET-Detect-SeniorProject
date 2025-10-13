#!/usr/bin/env python3
"""
RFID + Camera + API System for Raspberry Pi
ระบบสแกน RFID แล้วถ่ายภาพส่งไปยัง server
"""

import time
import requests
import json
import cv2
from mfrc522 import SimpleMFRC522
import RPi.GPIO as GPIO

# Configuration
API_URL = "https://pet-detect-seniorproject-production.up.railway.app/api/scan"
LED_PIN = 18  # LED indicator pin
BUZZER_PIN = 12  # Buzzer pin

class RFIDCameraSystem:
    def __init__(self):
        """Initialize RFID reader, camera, and GPIO"""
        self.rfid_reader = SimpleMFRC522()
        
        # Initialize USB Camera
        self.camera = cv2.VideoCapture(0)  # Use USB camera (device 0)
        if not self.camera.isOpened():
            print("❌ ไม่สามารถเปิด USB Camera ได้")
            # Try alternative camera index
            self.camera = cv2.VideoCapture(1)
            if not self.camera.isOpened():
                print("❌ ไม่พบ USB Camera")
                raise Exception("USB Camera not found")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        
        print("🔧 ระบบ RFID + USB Camera พร้อมใช้งาน")
        print("📋 วางบัตร RFID บนเครื่องอ่าน...")
    
    def scan_rfid(self):
        """Scan RFID card and return card ID"""
        try:
            print("🔍 กำลังสแกน RFID...")
            card_id, text = self.rfid_reader.read()
            print(f"✅ สแกนสำเร็จ! Card ID: {card_id}")
            return card_id
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการสแกน RFID: {e}")
            return None
    
    def capture_image(self):
        """Capture image using USB Camera"""
        try:
            timestamp = int(time.time())
            image_path = f"capture_{timestamp}.jpg"
            
            print("📸 กำลังถ่ายภาพ...")
            
            # Capture frame from USB camera
            ret, frame = self.camera.read()
            if not ret:
                print("❌ ไม่สามารถอ่านภาพจาก USB Camera ได้")
                return None
            
            # Save image
            cv2.imwrite(image_path, frame)
            print(f"✅ ถ่ายภาพสำเร็จ: {image_path}")
            
            return image_path
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการถ่ายภาพ: {e}")
            return None
    
    def send_to_api(self, image_path, card_id):
        """Send image and RFID data to API"""
        try:
            print("🌐 กำลังส่งข้อมูลไปยัง server...")
            
            # Prepare files and data
            files = {
                'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
            }
            
            data = {
                'card_id': str(card_id),
                'timestamp': int(time.time())
            }
            
            # Send POST request
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ ส่งข้อมูลสำเร็จ!")
                print(f"📊 ผลการวิเคราะห์:")
                print(f"   - ขวด: {result.get('bottle_count', 0)}")
                print(f"   - ฝา: {result.get('cap_count', 0)}")
                print(f"   - สลาก: {result.get('label_count', 0)}")
                print(f"   - คะแนน: {result.get('score', 0)}")
                
                # Visual feedback
                self.led_success()
                return result
            else:
                print(f"❌ เกิดข้อผิดพลาดในการส่งข้อมูล: {response.status_code}")
                self.led_error()
                return None
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
            self.led_error()
            return None
    
    def led_success(self):
        """Blink LED for success"""
        for _ in range(3):
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.2)
    
    def led_error(self):
        """Blink LED for error"""
        for _ in range(5):
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)
    
    def buzzer_beep(self):
        """Make buzzer sound"""
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
    
    def run_loop(self):
        """Main loop for RFID scanning and image capture"""
        print("\n🔄 เริ่มระบบ RFID + Camera Loop")
        print("📋 วางบัตร RFID บนเครื่องอ่านเพื่อเริ่มต้น...")
        
        while True:
            try:
                # Step 1: Scan RFID
                print("\n" + "="*50)
                print("🔍 รอการสแกน RFID...")
                card_id = self.scan_rfid()
                
                if card_id is None:
                    print("❌ ไม่สามารถสแกน RFID ได้ ลองใหม่อีกครั้ง")
                    time.sleep(2)
                    continue
                
                # Step 2: Capture image
                print("\n📸 กำลังถ่ายภาพ...")
                image_path = self.capture_image()
                
                if image_path is None:
                    print("❌ ไม่สามารถถ่ายภาพได้ ลองใหม่อีกครั้ง")
                    time.sleep(2)
                    continue
                
                # Step 3: Send to API
                print("\n🌐 กำลังส่งข้อมูลไปยัง server...")
                result = self.send_to_api(image_path, card_id)
                
                if result:
                    print("✅ กระบวนการเสร็จสิ้น!")
                    self.buzzer_beep()
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
    
    def cleanup(self):
        """Cleanup GPIO and camera"""
        GPIO.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()
        print("🧹 ทำความสะอาดระบบเสร็จสิ้น")

def main():
    """Main function"""
    system = RFIDCameraSystem()
    
    try:
        system.run_loop()
    finally:
        system.cleanup()

if __name__ == "__main__":
    main() 