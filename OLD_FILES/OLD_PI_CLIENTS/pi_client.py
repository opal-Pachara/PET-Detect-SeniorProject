"""
Raspberry Pi Client สำหรับเชื่อมต่อกับ API
รวม RFID + Camera + API Communication
"""

import requests
import cv2
import time
import json
import os
from mfrc522 import SimpleMFRC522
import RPi.GPIO as GPIO
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pet_detect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PETDetectClient:
    def __init__(self, api_url="http://localhost:5000"):
        """
        Initialize PET Detect Client
        
        Args:
            api_url (str): URL ของ API server
        """
        self.api_url = api_url.rstrip('/')
        self.rfid_reader = SimpleMFRC522()
        self.camera = None
        self.session = requests.Session()
        
        # กำหนด timeout สำหรับ API calls
        self.session.timeout = 30
        
        logger.info(f"🚀 PET Detect Client initialized")
        logger.info(f"🌐 API URL: {self.api_url}")
        
    def setup_camera(self, camera_index=0):
        """เปิดการใช้งาน USB Camera"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
                
            logger.info(f"📷 Camera {camera_index} opened successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Camera setup failed: {e}")
            return False
    
    def test_api_connection(self):
        """ทดสอบการเชื่อมต่อกับ API"""
        try:
            response = self.session.get(f"{self.api_url}/api/ping")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API connection successful: {data.get('message')}")
                return True
            else:
                logger.error(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            return False
    
    def capture_image(self, save_path=None):
        """ถ่ายรูปจาก camera"""
        if self.camera is None:
            logger.error("❌ Camera not initialized")
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.error("❌ Failed to capture image")
                return None
            
            # บันทึกรูปถ้าระบุ path
            if save_path:
                cv2.imwrite(save_path, frame)
                logger.info(f"💾 Image saved: {save_path}")
            
            logger.info("📸 Image captured successfully")
            return frame
        except Exception as e:
            logger.error(f"❌ Image capture failed: {e}")
            return None
    
    def send_image_to_api(self, image_data, image_path=None):
        """ส่งรูปภาพไปยัง API สำหรับวิเคราะห์"""
        try:
            # ถ้าเป็น numpy array ให้แปลงเป็น image file
            if image_path is None:
                # สร้างไฟล์ชั่วคราว
                temp_path = f"temp_image_{int(time.time())}.jpg"
                cv2.imwrite(temp_path, image_data)
                image_path = temp_path
                delete_temp = True
            else:
                delete_temp = False
            
            # เปิดไฟล์และส่งไป API
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                response = self.session.post(
                    f"{self.api_url}/api/scan", 
                    files=files
                )
            
            # ลบไฟล์ชั่วคราว
            if delete_temp and os.path.exists(image_path):
                os.remove(image_path)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API analysis successful")
                return data
            else:
                logger.error(f"❌ API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            return None
    
    def read_rfid(self, timeout=10):
        """อ่าน RFID card"""
        logger.info(f"🔍 Waiting for RFID card (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                card_id, text = self.rfid_reader.read_no_block()
                if card_id:
                    logger.info(f"💳 RFID detected - ID: {card_id}")
                    return card_id, text
            except Exception as e:
                logger.debug(f"RFID read attempt failed: {e}")
            
            time.sleep(0.1)
        
        logger.warning("⏰ RFID read timeout")
        return None, None
    
    def process_scan_result(self, result_data):
        """ประมวลผลและแสดงผลลัพธ์"""
        if not result_data or not result_data.get('success'):
            logger.error(f"❌ Analysis failed: {result_data.get('message') if result_data else 'No data'}")
            return False
        
        result = result_data.get('result', {})
        
        # แสดงผลลัพธ์
        print("\n" + "="*50)
        print("🎯 การวิเคราะห์เสร็จสิ้น")
        print("="*50)
        print(f"🍶 ขวด (Bottles): {result.get('bottle_count', 0)}")
        print(f"🥫 กระป๋อง (Cans): {result.get('can_count', 0)}")
        print(f"🧢 ฝา (Caps): {result.get('cap_count', 0)}")
        print(f"🏷️  สลาก (Labels): {result.get('label_count', 0)}")
        print(f"📊 คะแนนรวม: {result.get('score', 0)}")
        print(f"🔍 ตรวจพบทั้งหมด: {result.get('total_detections', 0)} รายการ")
        print("="*50)
        
        # บันทึกผลลัพธ์
        self.save_result(result)
        
        return True
    
    def save_result(self, result):
        """บันทึกผลลัพธ์ลงไฟล์"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'result': result
            }
            
            # บันทึกลง JSON file
            log_file = 'scan_results.json'
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            logger.info(f"💾 Result saved to {log_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save result: {e}")
    
    def run_continuous_scan(self):
        """รันระบบแบบต่อเนื่อง: RFID -> Camera -> API -> ผลลัพธ์"""
        logger.info("🔄 Starting continuous scan mode...")
        logger.info("🛑 Press Ctrl+C to stop")
        
        try:
            while True:
                print("\n" + "🔍 รอการสแกน RFID card...")
                
                # อ่าน RFID
                card_id, _ = self.read_rfid(timeout=30)
                
                if card_id:
                    print(f"✅ ตรวจพบ RFID card: {card_id}")
                    
                    # ถ่ายรูป
                    print("📸 กำลังถ่ายรูป...")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"captured_images/scan_{timestamp}.jpg"
                    
                    # สร้างโฟลเดอร์ถ้าไม่มี
                    os.makedirs("captured_images", exist_ok=True)
                    
                    image = self.capture_image(save_path=image_path)
                    
                    if image is not None:
                        print("🤖 กำลังวิเคราะห์ด้วย AI...")
                        
                        # ส่งไป API
                        result = self.send_image_to_api(image, image_path)
                        
                        # แสดงผลลัพธ์
                        self.process_scan_result(result)
                    else:
                        print("❌ การถ่ายรูปล้มเหลว")
                    
                    # รอก่อนรอบถัดไป
                    print("\n⏱️  รอ 3 วินาทีก่อนรอบถัดไป...")
                    time.sleep(3)
                else:
                    print("⏰ ไม่พบ RFID card ภายในเวลาที่กำหนด")
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping continuous scan...")
        except Exception as e:
            logger.error(f"❌ Error in continuous scan: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """ปิดการใช้งาน resources"""
        try:
            if self.camera:
                self.camera.release()
                logger.info("📷 Camera released")
            
            GPIO.cleanup()
            logger.info("🔌 GPIO cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

def main():
    # ใส่ URL ของ API server ที่นี่
    API_URL = "http://localhost:5000"  # หรือ IP ของเครื่องที่รัน API
    
    client = PETDetectClient(api_url=API_URL)
    
    # Setup
    if not client.setup_camera():
        print("❌ ไม่สามารถเปิด camera ได้")
        return
    
    if not client.test_api_connection():
        print("❌ ไม่สามารถเชื่อมต่อกับ API ได้")
        return
    
    # เริ่มการทำงาน
    client.run_continuous_scan()

if __name__ == "__main__":
    main()