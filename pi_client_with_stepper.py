"""
PET Detect Client with Stepper Motor (Micro Step)
ใช้ Stepper Motor แทน DC Motor เพื่อความแม่นยำสูง
- ขวด (Bottle) → หมุนซ้าย 90°
- กระป๋อง (Can) → หมุนขวา 90°
- หมุนกลับตำแหน่งเดิมหลังเสร็จ
"""

import requests
import requests.exceptions
import cv2
import time
import json
import os
from mfrc522 import SimpleMFRC522
import RPi.GPIO as GPIO
from datetime import datetime
import logging
from stepper_motor_controller import StepperMotorController

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

class PETDetectClientWithStepper:
    def __init__(self, api_url="http://192.168.1.31:5000"):
        """
        Initialize PET Detect Client with Stepper Motor Control
        """
        self.api_url = api_url.rstrip('/')
        self.rfid_reader = SimpleMFRC522()
        self.camera = None
        # Initialize Stepper Motor (ไม่ใช้ ENA pin)
        self.stepper = StepperMotorController(
            step_pin=18,    # PUL+ → GPIO 18 (Pin 12)
            dir_pin=19,     # DIR+ → GPIO 19 (Pin 35)  
            enable_pin=None # ENA+ → ไม่ต้องต่อ (มอเตอร์เปิดใช้งานอยู่เสมอ)
        )
        self.session = requests.Session()
        self.session.timeout = 5
        
        logger.info(f"PET Detect Client with Stepper Motor initialized")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Stepper Control: LEFT 90° (Bottle), RIGHT 90° (Can)")
    
    def test_api_connection(self):
        """ทดสอบการเชื่อมต่อกับ API"""
        try:
            print(f"ทดสอบการเชื่อมต่อไปยัง: {self.api_url}/api/ping")
            print("กำลังทดสอบ (timeout 5 วินาที)...")
            
            response = self.session.get(f"{self.api_url}/api/ping", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API connection successful: {data.get('message')}")
                print("เชื่อมต่อ API สำเร็จ!")
                return True
            else:
                logger.error(f"API returned status {response.status_code}")
                print(f"API ตอบกลับ status: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print("API connection timeout (5 วินาที)")
            print("กรุณาตรวจสอบ API บนเครื่อง Windows")
            return False
        except requests.exceptions.ConnectionError:
            print("ไม่สามารถเชื่อมต่อไปยัง API ได้")
            print("กรุณาตรวจสอบ IP address และ network")
            return False
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            print(f"เกิดข้อผิดพลาด: {e}")
            return False
    
    def open_camera(self, camera_index=0, timeout=5):
        """เปิดกล้องเฉพาะเมื่อต้องการใช้งาน"""
        try:
            print("เปิดกล้อง...")
            start_time = time.time()
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # รอให้กล้องพร้อม
            for i in range(5):
                ret, frame = self.camera.read()
                if ret:
                    break
                time.sleep(0.5)
                
                if time.time() - start_time > timeout:
                    raise Exception("Camera timeout")
            
            if not ret:
                raise Exception("Camera not ready")
                
            logger.info(f"Camera opened successfully in {time.time() - start_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Camera open failed: {e}")
            self.close_camera()
            return False
    
    def close_camera(self):
        """ปิดกล้องหลังใช้งานเสร็จ"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                logger.info("Camera closed")
        except Exception as e:
            logger.error(f"Camera close error: {e}")
    
    def capture_image_quick(self, save_path=None, max_attempts=3):
        """ถ่ายรูปอย่างรวดเร็ว"""
        if self.camera is None:
            logger.error("Camera not opened")
            return None
            
        try:
            for attempt in range(max_attempts):
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    if save_path:
                        cv2.imwrite(save_path, frame)
                        logger.info(f"Image saved: {save_path}")
                    
                    logger.info(f"Image captured successfully (attempt {attempt + 1})")
                    return frame
                
                logger.warning(f"Capture attempt {attempt + 1} failed, retrying...")
                time.sleep(0.2)
            
            logger.error("All capture attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
            return None
    
    def send_image_to_api(self, image_data, image_path=None):
        """ส่งรูปภาพไปยัง API สำหรับวิเคราะห์"""
        try:
            if image_path is None:
                temp_path = f"temp_image_{int(time.time())}.jpg"
                cv2.imwrite(temp_path, image_data)
                image_path = temp_path
                delete_temp = True
            else:
                delete_temp = False
            
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                response = self.session.post(
                    f"{self.api_url}/api/scan", 
                    files=files
                )
            
            if delete_temp and os.path.exists(image_path):
                os.remove(image_path)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API analysis successful")
                return data
            else:
                logger.error(f"API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def read_rfid_with_timeout(self, timeout=30):
        """อ่าน RFID card พร้อม timeout"""
        print(f"🔍 รอการสแกน RFID card (timeout: {timeout} วินาที)...")
        print("📱 วางบัตร RFID ใกล้ตัวอ่าน...")
        logger.info(f"Waiting for RFID card (timeout: {timeout}s)...")
        
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < timeout:
            try:
                # ลองหลายวิธี RFID reading
                card_id = None
                text = ""
                
                # วิธีที่ 1: read_no_block
                try:
                    card_id, text = self.rfid_reader.read_no_block()
                except:
                    pass
                
                # วิธีที่ 2: read_id_no_block (ถ้าวิธีแรกไม่ได้)
                if not card_id:
                    try:
                        card_id = self.rfid_reader.read_id_no_block()
                        text = ""
                    except:
                        pass
                
                if card_id:
                    print(f"✅ RFID detected - ID: {card_id}")
                    logger.info(f"RFID detected - ID: {card_id}")
                    return card_id, text
                
                # แสดงสถานะทุก 5 วินาที
                current_time = time.time()
                if current_time - last_status_time >= 5:
                    elapsed = int(current_time - start_time)
                    remaining = timeout - elapsed
                    print(f"⏳ รอ RFID... เหลือ {remaining} วินาที")
                    last_status_time = current_time
                
            except Exception as e:
                logger.debug(f"RFID read attempt failed: {e}")
            
            time.sleep(0.2)  # ลด delay ลงเล็กน้อย
        
        print("❌ RFID timeout - ไม่พบบัตร")
        print("🔧 แนะนำการแก้ปัญหา:")
        print("   1. ตรวจสอบการต่อสาย RFID")
        print("   2. วางบัตรใกล้ตัวอ่านมากขึ้น")
        print("   3. ลองใช้บัตร RFID อื่น")
        print("   4. รัน: python test_rfid_connection.py")
        logger.warning("RFID read timeout")
        return None, None
    
    def control_stepper_by_detection(self, result_data):
        """ควบคุม Stepper Motor ตามผลการตรวจจับ และหมุนกลับมาที่เดิม"""
        if not result_data or not result_data.get('success'):
            logger.warning("No valid detection result for stepper control")
            return
        
        result = result_data.get('result', {})
        bottle_count = result.get('bottle_count', 0)
        can_count = result.get('can_count', 0)
        
        print("\nการควบคุม Stepper Motor:")
        print("="*35)
        print(f"📐 ตำแหน่งปัจจุบัน: {self.stepper.get_position_degrees()}°")
        
        if bottle_count > 0 and can_count > 0:
            # มีทั้งขวดและกระป๋อง - ให้ความสำคัญกับกระป๋อง (คะแนนสูงกว่า)
            print(f"พบทั้งขวด ({bottle_count}) และกระป๋อง ({can_count})")
            print("เลือกหมุนขวา 120° (กระป๋อง - คะแนนสูงกว่า)")
            self.stepper.rotate_right(120, speed=1000)
            
        elif bottle_count > 0:
            # พบขวดเท่านั้น
            print(f"พบขวด ({bottle_count} อัน)")
            print("หมุนซ้าย 90°")
            self.stepper.rotate_left(90, speed=1000)
            
        elif can_count > 0:
            # พบกระป๋องเท่านั้น
            print(f"พบกระป๋อง ({can_count} อัน)")
            print("หมุนขวา 90°")
            self.stepper.rotate_right(90, speed=1000)
            
        else:
            # ไม่พบอะไร
            print("ไม่พบขวดหรือกระป๋อง")
            print("ไม่หมุน Stepper Motor")
            print("="*35)
            return
        
        # รอสักครู่แล้วหมุนกลับ
        print("รอ 2 วินาที...")
        time.sleep(2)
        
        print("หมุนกลับตำแหน่งเริ่มต้น...")
        self.stepper.return_to_home(speed=1200)
        
        print(f"📐 ตำแหน่งสุดท้าย: {self.stepper.get_position_degrees()}°")
        print("การควบคุม Stepper Motor เสร็จสิ้น")
        print("="*35)
    
    def process_scan_result(self, result_data):
        """ประมวลผลและแสดงผลลัพธ์"""
        if not result_data or not result_data.get('success'):
            logger.error(f"Analysis failed: {result_data.get('message') if result_data else 'No data'}")
            return False
        
        result = result_data.get('result', {})
        
        # แสดงผลลัพธ์
        print("\n" + "="*50)
        print("การวิเคราะห์เสร็จสิ้น")
        print("="*50)
        print(f"ขวด (Bottles): {result.get('bottle_count', 0)}")
        print(f"กระป๋อง (Cans): {result.get('can_count', 0)}")
        print(f"ฝา (Caps): {result.get('cap_count', 0)}")
        print(f"สลาก (Labels): {result.get('label_count', 0)}")
        print(f"คะแนนรวม: {result.get('score', 0)}")
        print(f"ตรวจพบทั้งหมด: {result.get('total_detections', 0)} รายการ")
        print("="*50)
        
        # ควบคุม Stepper Motor ตามผลการตรวจจับ
        self.control_stepper_by_detection(result_data)
        
        # บันทึกผลลัพธ์
        self.save_result(result)
        return True
    
    def save_result(self, result):
        """บันทึกผลลัพธ์ลงไฟล์"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'result': result,
                'stepper_position': self.stepper.get_position_degrees()
            }
            
            log_file = 'scan_results.json'
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Result saved to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    def run_stepper_scan_system(self):
        """รันระบบแบบมี Stepper Motor: RFID → Camera → API → Stepper Control"""
        logger.info("Starting PET Detect with Stepper Motor Control...")
        logger.info("Flow: RFID → Camera → API → Stepper → Result")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                print("\n" + "รอการสแกน RFID card...")
                
                # ขั้นตอน 1: รอ RFID
                card_id, _ = self.read_rfid_with_timeout(timeout=30)
                
                if card_id:
                    print(f"ตรวจพบ RFID card: {card_id}")
                    
                    # ขั้นตอน 2: เปิดกล้อง
                    if self.open_camera():
                        try:
                            # ขั้นตอน 3: ถ่ายรูป
                            print("กำลังถ่ายรูป...")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = f"captured_images/scan_{timestamp}.jpg"
                            
                            os.makedirs("captured_images", exist_ok=True)
                            
                            image = self.capture_image_quick(save_path=image_path)
                            
                            if image is not None:
                                # ขั้นตอน 4: วิเคราะห์ด้วย AI
                                print("กำลังวิเคราะห์ด้วย AI...")
                                result = self.send_image_to_api(image, image_path)
                                
                                # ขั้นตอน 5: แสดงผลลัพธ์และควบคุม Stepper
                                self.process_scan_result(result)
                            else:
                                print("การถ่ายรูปล้มเหลว")
                                
                        finally:
                            # ขั้นตอน 6: ปิดกล้องเสมอ
                            self.close_camera()
                            
                    else:
                        print("ไม่สามารถเปิดกล้องได้")
                    
                    # รอก่อนรอบถัดไป
                    print("\nรอ 5 วินาทีก่อนรอบถัดไป...")
                    time.sleep(5)
                    
                else:
                    print("ไม่พบ RFID card ภายในเวลาที่กำหนด")
                    
        except KeyboardInterrupt:
            logger.info("\nStopping stepper scan system...")
        except Exception as e:
            logger.error(f"Error in stepper scan system: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """ปิดการใช้งาน resources"""
        try:
            self.close_camera()
            self.stepper.cleanup()
            GPIO.cleanup()
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    API_URL = "http://192.168.1.31:5000"
    
    client = PETDetectClientWithStepper(api_url=API_URL)
    
    # ทดสอบการเชื่อมต่อ API
    if not client.test_api_connection():
        print("ไม่สามารถเชื่อมต่อกับ API ได้")
        return
    
    # เริ่มการทำงานระบบที่มี Stepper Motor
    client.run_stepper_scan_system()

if __name__ == "__main__":
    main()