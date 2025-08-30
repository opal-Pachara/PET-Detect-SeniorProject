#!/usr/bin/env python3
"""
PET Detect Client - ใช้ subprocess สำหรับ RFID
เพื่อหลีกเลี่ยง GPIO conflict
"""

import time
import cv2
import logging
import requests
import signal
import sys
import subprocess
import os
import json
from stepper_motor_controller import StepperMotorController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://192.168.1.31:5000"

class PETDetectSubprocess:
    def __init__(self, api_url=API_URL):
        self.api_url = api_url.rstrip('/')
        self.running = True
        self.camera = None
        self.stepper = None
        
        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Session for API
        self.session = requests.Session()
        self.session.timeout = 5
        
        # สร้าง RFID script helper
        self.create_rfid_helper()
        
        # Initialize Stepper Motor
        try:
            print("🔧 กำลัง initialize Stepper Motor...")
            self.stepper = StepperMotorController(
                step_pin=18,
                dir_pin=19,
                enable_pin=None
            )
            print("✅ Stepper Motor พร้อมใช้งาน")
        except Exception as e:
            print(f"❌ Stepper Error: {e}")
            self.stepper = None
    
    def signal_handler(self, signum, frame):
        print("\n🛑 กำลังหยุดระบบ...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def create_rfid_helper(self):
        """สร้าง RFID helper script"""
        rfid_script = """#!/usr/bin/env python3
import time
import json
import sys
from mfrc522 import SimpleMFRC522

def read_rfid_once():
    try:
        reader = SimpleMFRC522()
        card_id, text = reader.read_no_block()
        
        if card_id:
            result = {
                'success': True,
                'card_id': card_id,
                'text': text.strip() if text else ''
            }
        else:
            result = {'success': False}
        
        print(json.dumps(result))
        
    except Exception as e:
        result = {'success': False, 'error': str(e)}
        print(json.dumps(result))

if __name__ == "__main__":
    read_rfid_once()
"""
        
        with open('rfid_helper.py', 'w') as f:
            f.write(rfid_script)
        
        # Make executable
        os.chmod('rfid_helper.py', 0o755)
    
    def read_rfid_subprocess(self, timeout=30):
        """อ่าน RFID ผ่าน subprocess"""
        print(f"🔍 รอการสแกน RFID card (timeout: {timeout} วินาที)...")
        print("📱 วางบัตร RFID ใกล้ตัวอ่าน...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and self.running:
            try:
                # รัน RFID helper script
                result = subprocess.run(
                    ['python3', 'rfid_helper.py'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout.strip())
                        if data.get('success'):
                            card_id = data.get('card_id')
                            text = data.get('text', '')
                            print(f"✅ RFID detected - ID: {card_id}")
                            return card_id, text
                    except json.JSONDecodeError:
                        pass
                
                time.sleep(0.2)
                
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                logger.debug(f"Subprocess error: {e}")
                time.sleep(0.5)
        
        print("❌ RFID timeout")
        return None, None
    
    def open_camera(self):
        """เปิดกล้อง"""
        try:
            print("📸 เปิดกล้อง...")
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                print("❌ ไม่สามารถเปิดกล้องได้")
                return False
            
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
    
    def control_stepper(self, result_data):
        """ควบคุม Stepper Motor"""
        if not self.stepper:
            print("❌ Stepper motor not available")
            return
        
        if not result_data or not result_data.get('success'):
            return
        
        result = result_data.get('result', {})
        bottle_count = result.get('bottle_count', 0)
        can_count = result.get('can_count', 0)
        
        print("\nการควบคุม Stepper Motor:")
        print("=" * 35)
        
        try:
            if bottle_count > 0 and can_count > 0:
                print(f"พบทั้งขวด ({bottle_count}) และกระป๋อง ({can_count})")
                print("หมุนขวา 90° (ช้า ๆ)")
                self.stepper.move_degrees(90, speed=400)  # หมุนขวา 90°
                time.sleep(2)
                
            elif bottle_count > 0:
                print(f"พบขวด ({bottle_count} อัน)")
                print("หมุนซ้าย 90° (ช้า ๆ)")
                self.stepper.move_degrees(-90, speed=400)  # หมุนซ้าย 90°
                time.sleep(2)
                
            elif can_count > 0:
                print(f"พบกระป๋อง ({can_count} อัน)")
                print("หมุนขวา 90° (ช้า ๆ)")
                self.stepper.move_degrees(90, speed=400)  # หมุนขวา 90°
                time.sleep(2)
            else:
                print("ไม่พบขวดหรือกระป๋อง")
                return
            
            # กลับตำแหน่งเดิม
            print("หมุนกลับตำแหน่งเริ่มต้น (ช้า ๆ)...")
            self.stepper.return_to_home(speed=500)  # ความเร็วกลับบ้านช้า
            time.sleep(1)
            
            print("✅ ควบคุม Stepper Motor เสร็จสิ้น")
            
        except Exception as e:
            print(f"❌ Stepper control error: {e}")
        
        print("=" * 35)
    
    def run_single_scan(self):
        """รันการสแกนครั้งเดียว"""
        print("\n🚀 เริ่มกระบวนการสแกน PET")
        print("=" * 50)
        
        # 1. สแกน RFID ผ่าน subprocess
        card_id, text = self.read_rfid_subprocess(timeout=30)
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
        self.control_stepper(result)
        
        print("✅ การสแกนเสร็จสิ้น!")
        return True
    
    def run_continuous_scan_system(self):
        """รันระบบสแกนต่อเนื่อง"""
        print("🚀 ระบบสแกน PET (Subprocess RFID)")
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
            
            # ลบ helper script
            if os.path.exists('rfid_helper.py'):
                os.remove('rfid_helper.py')
            
            print("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function"""
    print("🔧 PET Detect System - Subprocess RFID Method")
    print("=" * 60)
    
    client = PETDetectSubprocess(api_url=API_URL)
    
    print("✅ System พร้อมใช้งาน!")
    print("\n📋 วิธีการทำงาน:")
    print("- RFID: ใช้ subprocess (แยก process)")
    print("- Camera: ภายใน main process")
    print("- Stepper: ภายใน main process")
    print("- API: ภายใน main process")
    
    client.run_continuous_scan_system()
    client.cleanup()

if __name__ == "__main__":
    main()