#!/usr/bin/env python3
"""
ทดสอบ RFID + Camera เท่านั้น (ไม่มี Stepper Motor)
เพื่อแยกปัญหา
"""

import time
import cv2
import logging
import requests
import signal
import sys
from mfrc522 import SimpleMFRC522

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://192.168.1.31:5000"

class PETDetectNoStepper:
    def __init__(self, api_url=API_URL):
        self.api_url = api_url.rstrip('/')
        self.running = True
        self.camera = None
        
        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Session for API
        self.session = requests.Session()
        self.session.timeout = 5
        
        # Initialize RFID
        try:
            print("🔧 กำลัง initialize RFID...")
            self.rfid_reader = SimpleMFRC522()
            print("✅ RFID พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ RFID Error: {e}")
            self.rfid_reader = None
    
    def signal_handler(self, signum, frame):
        print("\n🛑 กำลังหยุดระบบ...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def read_rfid_simple(self, timeout=30):
        """อ่าน RFID แบบง่าย"""
        if not self.rfid_reader:
            return None, None
            
        print(f"🔍 รอการสแกน RFID card (timeout: {timeout} วินาที)...")
        print("📱 วางบัตร RFID ใกล้ตัวอ่าน...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and self.running:
            try:
                card_id, text = self.rfid_reader.read_no_block()
                if card_id:
                    print(f"✅ RFID detected - ID: {card_id}")
                    return card_id, text
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"RFID read failed: {e}")
                time.sleep(0.5)
        
        print("❌ RFID timeout")
        return None, None
    
    def test_camera(self):
        """ทดสอบกล้อง"""
        try:
            print("📸 ทดสอบกล้อง...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ ไม่สามารถเปิดกล้องได้")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                cv2.imwrite("test_image.jpg", frame)
                print("✅ กล้องทำงานได้ - บันทึก test_image.jpg")
                return True
            else:
                print("❌ ไม่สามารถถ่ายภาพได้")
                return False
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def test_api(self):
        """ทดสอบ API"""
        try:
            print("🤖 ทดสอบ API connection...")
            response = self.session.get(f"{self.api_url}/api/ping", timeout=5)
            if response.status_code == 200:
                print("✅ API ทำงานได้")
                return True
            else:
                print(f"❌ API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Error: {e}")
            return False
    
    def run_simple_test(self):
        """รันทดสอบง่าย ๆ"""
        print("🚀 ทดสอบระบบแบบง่าย (ไม่มี Stepper)")
        print("=" * 50)
        
        # ทดสอบ components
        print("\n📋 ทดสอบ Components:")
        
        # Test API
        api_ok = self.test_api()
        
        # Test Camera  
        camera_ok = self.test_camera()
        
        if not api_ok or not camera_ok:
            print("❌ บาง component ไม่ทำงาน")
            return
        
        print("\n✅ ทุก component พร้อม!")
        
        # รันลูปหลัก
        scan_count = 0
        while self.running:
            try:
                scan_count += 1
                print(f"\n🔄 รอบที่ {scan_count}:")
                
                # 1. สแกน RFID
                card_id, text = self.read_rfid_simple(timeout=30)
                if not card_id:
                    continue
                
                # 2. ถ่ายภาพ
                print("📸 เปิดกล้องและถ่ายภาพ...")
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            image_path = f"scan_{scan_count}.jpg"
                            cv2.imwrite(image_path, frame)
                            print(f"✅ บันทึกภาพ: {image_path}")
                            
                            # 3. ส่งไปยัง API
                            print("🤖 ส่งไปยัง API...")
                            try:
                                with open(image_path, 'rb') as f:
                                    files = {'image': f}
                                    response = self.session.post(
                                        f"{self.api_url}/api/scan",
                                        files=files,
                                        timeout=10
                                    )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    print("✅ ได้รับผลจาก API:")
                                    
                                    if result.get('success'):
                                        data = result.get('result', {})
                                        print(f"   ขวด: {data.get('bottle_count', 0)}")
                                        print(f"   กระป๋อง: {data.get('can_count', 0)}")
                                        print(f"   คะแนน: {data.get('score', 0)}")
                                    else:
                                        print("   ไม่พบวัตถุ")
                                else:
                                    print(f"❌ API Error: {response.status_code}")
                            except Exception as e:
                                print(f"❌ API Error: {e}")
                        else:
                            print("❌ ไม่สามารถถ่ายภาพได้")
                    else:
                        print("❌ ไม่สามารถเปิดกล้องได้")
                except Exception as e:
                    print(f"❌ Camera Error: {e}")
                
                print("💤 รอ 3 วินาทีก่อนรอบต่อไป...")
                time.sleep(3)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """ทำความสะอาด"""
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        print("🧹 Cleanup completed")

def main():
    client = PETDetectNoStepper()
    if client.rfid_reader:
        client.run_simple_test()
    else:
        print("❌ ไม่สามารถเริ่มต้น RFID ได้")

if __name__ == "__main__":
    main()