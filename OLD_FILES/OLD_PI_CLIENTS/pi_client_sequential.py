#!/usr/bin/env python3
"""
PET Detect Client - แก้ปัญหา GPIO conflict
โดยแยก Initialization ออกจากกัน
"""

import time
import cv2
import logging
import requests
import signal
import sys
import os
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API URL (เปลี่ยนตาม IP ของ Windows)
API_URL = "http://192.168.1.31:5000"

class PETDetectClientSequential:
    def __init__(self, api_url=API_URL):
        """
        Initialize PET Detect Client - Sequential GPIO Setup
        """
        self.api_url = api_url.rstrip('/')
        self.camera = None
        self.rfid_reader = None
        self.stepper = None
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        self.running = True
        
        # Session for API calls
        self.session = requests.Session()
        self.session.timeout = 5
        
        logger.info(f"PET Detect Client Sequential initialized")
        logger.info(f"API URL: {self.api_url}")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 กำลังหยุดระบบ...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def init_rfid_only(self):
        """Initialize RFID เท่านั้น"""
        try:
            print("🔧 กำลัง initialize RFID...")
            from mfrc522 import SimpleMFRC522
            self.rfid_reader = SimpleMFRC522()
            logger.info("RFID reader initialized successfully")
            print("✅ RFID พร้อมใช้งาน")
            return True
        except Exception as e:
            logger.error(f"RFID initialization failed: {e}")
            print(f"❌ RFID Error: {e}")
            return False
    
    def init_stepper_only(self):
        """Initialize Stepper เท่านั้น"""
        try:
            print("🔧 กำลัง initialize Stepper Motor...")
            from stepper_motor_controller import StepperMotorController
            self.stepper = StepperMotorController(
                step_pin=18,    # PUL+ → GPIO 18 (Pin 12)
                dir_pin=19,     # DIR+ → GPIO 19 (Pin 35)  
                enable_pin=None # ENA+ → ไม่ต้องต่อ
            )
            logger.info("Stepper motor initialized successfully")
            print("✅ Stepper Motor พร้อมใช้งาน")
            return True
        except Exception as e:
            logger.error(f"Stepper motor initialization failed: {e}")
            print(f"❌ Stepper Error: {e}")
            return False
    
    def cleanup_gpio(self):
        """ทำความสะอาด GPIO"""
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
            print("🧹 GPIO cleaned up")
        except:
            pass
    
    def read_rfid_simple(self, timeout=30):
        """อ่าน RFID แบบง่าย"""
        if not self.rfid_reader:
            print("❌ RFID reader not initialized")
            return None, None
            
        print(f"🔍 รอการสแกน RFID card (timeout: {timeout} วินาที)...")
        print("📱 วางบัตร RFID ใกล้ตัวอ่าน...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and self.running:
            try:
                # ใช้วิธีเดียว - read_no_block
                card_id, text = self.rfid_reader.read_no_block()
                
                if card_id:
                    print(f"✅ RFID detected - ID: {card_id}")
                    logger.info(f"RFID detected - ID: {card_id}")
                    return card_id, text
                
                time.sleep(0.1)  # delay สั้น
                
            except Exception as e:
                logger.debug(f"RFID read failed: {e}")
                time.sleep(0.5)
        
        print("❌ RFID timeout")
        return None, None
    
    def open_camera(self):
        """เปิดกล้อง USB"""
        try:
            print("📸 เปิดกล้อง...")
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                print("❌ ไม่สามารถเปิดกล้องได้")
                return False
            
            # ตั้งค่ากล้อง
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("✅ เปิดกล้องสำเร็จ")
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def capture_image(self):
        """ถ่ายภาพ"""
        if not self.camera:
            print("❌ กล้องไม่ได้เปิด")
            return None
        
        try:
            print("📷 ถ่ายภาพ...")
            ret, frame = self.camera.read()
            
            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
                print(f"✅ บันทึกภาพ: {image_path}")
                return image_path
            else:
                print("❌ ไม่สามารถถ่ายภาพได้")
                return None
        except Exception as e:
            print(f"❌ Capture error: {e}")
            return None
    
    def close_camera(self):
        """ปิดกล้อง"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("📸 ปิดกล้องแล้ว")
    
    def send_image_to_api(self, image_path):
        """ส่งภาพไปยัง API"""
        try:
            print("🤖 ส่งภาพไปยัง AI API...")
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = self.session.post(
                    f"{self.api_url}/api/scan",
                    files=files,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ ได้รับผลจาก API")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def process_scan_result(self, result_data):
        """แสดงผลการวิเคราะห์"""
        if not result_data or not result_data.get('success'):
            print("❌ ไม่ได้รับผลการวิเคราะห์")
            return
        
        result = result_data.get('result', {})
        
        print("\nการวิเคราะห์ AI:")
        print("=" * 40)
        print(f"ขวด (Bottles): {result.get('bottle_count', 0)}")
        print(f"กระป๋อง (Cans): {result.get('can_count', 0)}")
        print(f"ฝา (Caps): {result.get('cap_count', 0)}")
        print(f"สลาก (Labels): {result.get('label_count', 0)}")
        print(f"จำนวนรวม: {result.get('total_detections', 0)}")
        print(f"คะแนน: {result.get('score', 0)}")
        print("=" * 40)
    
    def control_stepper_simple(self, result_data):
        """ควบคุม Stepper Motor แบบง่าย"""
        if not self.stepper:
            print("❌ Stepper motor not available")
            return
        
        if not result_data or not result_data.get('success'):
            print("❌ ไม่มีข้อมูลสำหรับควบคุม motor")
            return
        
        result = result_data.get('result', {})
        bottle_count = result.get('bottle_count', 0)
        can_count = result.get('can_count', 0)
        
        print("\nการควบคุม Stepper Motor:")
        print("=" * 35)
        
        try:
            if bottle_count > 0 and can_count > 0:
                print(f"พบทั้งขวด ({bottle_count}) และกระป๋อง ({can_count})")
                print("หมุนขวา 120° (กระป๋อง - คะแนนสูงกว่า)")
                self.stepper.move_degrees(120, step_delay=0.0005)
                time.sleep(2)
                
            elif bottle_count > 0:
                print(f"พบขวด ({bottle_count} อัน)")
                print("หมุนซ้าย 90°")
                self.stepper.move_degrees(-90, step_delay=0.0005)
                time.sleep(2)
                
            elif can_count > 0:
                print(f"พบกระป๋อง ({can_count} อัน)")
                print("หมุนขวา 90°")
                self.stepper.move_degrees(90, step_delay=0.0005)
                time.sleep(2)
            else:
                print("ไม่พบขวดหรือกระป๋อง - ไม่หมุน motor")
                return
            
            # กลับตำแหน่งเดิม
            print("หมุนกลับตำแหน่งเริ่มต้น...")
            self.stepper.return_to_home(step_delay=0.0005)
            time.sleep(1)
            
            print("✅ การควบคุม Stepper Motor เสร็จสิ้น")
            
        except Exception as e:
            print(f"❌ Stepper control error: {e}")
        
        print("=" * 35)
    
    def run_single_scan(self):
        """รันการสแกนครั้งเดียว"""
        print("\n🚀 เริ่มกระบวนการสแกน PET")
        print("=" * 50)
        
        # 1. สแกน RFID
        card_id, text = self.read_rfid_simple(timeout=30)
        if not card_id:
            print("❌ ไม่พบบัตร RFID")
            return False
        
        # 2. เปิดกล้องและถ่ายภาพ
        if not self.open_camera():
            return False
        
        image_path = self.capture_image()
        self.close_camera()
        
        if not image_path:
            return False
        
        # 3. ส่งไปยัง API
        result = self.send_image_to_api(image_path)
        if not result:
            return False
        
        # 4. แสดงผล
        self.process_scan_result(result)
        
        # 5. ควบคุม motor
        self.control_stepper_simple(result)
        
        print("✅ การสแกนเสร็จสิ้น!")
        return True
    
    def run_continuous_scan_system(self):
        """รันระบบสแกนต่อเนื่อง"""
        print("🚀 ระบบสแกน PET แบบต่อเนื่อง")
        print("กด Ctrl+C เพื่อหยุด")
        print("=" * 60)
        
        scan_count = 0
        
        while self.running:
            try:
                scan_count += 1
                print(f"\n🔄 รอบที่ {scan_count}:")
                
                success = self.run_single_scan()
                
                if success:
                    print("💤 รอ 3 วินาทีก่อนรอบต่อไป...")
                    time.sleep(3)
                else:
                    print("💤 รอ 2 วินาทีก่อนลองใหม่...")
                    time.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """ทำความสะอาด"""
        try:
            if self.camera:
                self.close_camera()
            if self.stepper:
                self.stepper.cleanup()
            self.cleanup_gpio()
            print("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function"""
    print("🔧 PET Detect System - Sequential GPIO Initialization")
    print("=" * 70)
    
    # Create client
    client = PETDetectClientSequential(api_url=API_URL)
    
    # Initialize components sequentially
    print("📋 กำลัง initialize components...")
    
    # 1. Initialize RFID first
    if not client.init_rfid_only():
        print("❌ ไม่สามารถ initialize RFID ได้")
        return
    
    time.sleep(1)  # หน่วงเวลา
    
    # 2. Initialize Stepper
    if not client.init_stepper_only():
        print("❌ ไม่สามารถ initialize Stepper ได้")
        return
    
    print("\n✅ ทุก component พร้อมแล้ว!")
    print("\n📋 ขั้นตอนการทำงาน:")
    print("1. สแกน RFID card")
    print("2. ถ่ายภาพด้วย USB camera")
    print("3. ส่งภาพไปยัง AI API")
    print("4. ควบคุม Stepper Motor ตามผลลัพธ์")
    print("5. กลับตำแหน่งเริ่มต้น")
    
    # Run continuous system
    client.run_continuous_scan_system()
    
    # Cleanup
    client.cleanup()

if __name__ == "__main__":
    main()