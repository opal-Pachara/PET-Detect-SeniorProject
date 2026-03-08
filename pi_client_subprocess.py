#!/usr/bin/env python3
"""
PET Detect - Raspberry Pi Client
ระบบสแกนขวด PET บน Raspberry Pi

หน้าที่หลัก:
- อ่านบัตร RFID (ผ่าน subprocess แยกเพื่อหลีกเลี่ยง GPIO conflict)
- ถ่ายภาพด้วยกล้อง USB
- ส่งภาพไป Cloud API (YOLOv11) วิเคราะห์
- แสดงผลบน LCD, LED, Buzzer
- บันทึกคะแนนไปยัง Member System
- ควบคุม Stepper Motor คัดแยกขวด/กระป๋อง
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
import atexit
import RPi.GPIO as GPIO
from stepper_motor_controller import StepperMotorController


GPIO.setwarnings(False)

# Import LCD Display
try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False
    print("LCD disable")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ ค่าคงที่ ============
API_URL = "https://pet-detect-ai-api.onrender.com"  # Cloud AI API สำหรับวิเคราะห์ภาพ


def _silence_buzzer_on_exit():
    """เมื่อโปรแกรมจบ ให้ Buzzer เงียบ (Active-LOW: HIGH=เงียบ)"""
    try:
        subprocess.run(['raspi-gpio', 'set', '17', 'op', 'dh'], timeout=2, capture_output=True)
    except Exception:
        pass


atexit.register(_silence_buzzer_on_exit)


# ============ คลาสหลัก ============
class PETDetectSubprocess:
    """คลาสควบคุมระบบสแกน PET บน Raspberry Pi"""
    def __init__(self, api_url=API_URL):
        self.api_url = api_url.rstrip('/')
        self.running = True
        self.camera = None
        self.stepper = None
        self.led_pin = 4   # GPIO 4 สำหรับ LED
        self.buzzer_pin = 17   # GPIO 17 สำหรับ Buzzer (VCC/GND/I/O)
        self.lcd = None    
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.session = requests.Session()
        self.session.timeout = 5
        
        # สร้าง สคริป RFID
        self.create_rfid_helper()
        
        self.setup_led()
        self.setup_buzzer()
        
        # Stepper Motor
        try:
            self.stepper = StepperMotorController(
                step_pin=18,
                dir_pin=19,
                enable_pin=None
            )
        except Exception as e:
            print(f"Stepper Error: {e}")
            self.stepper = None
    
    def setup_led(self):
        """ตั้งค่า GPIO ไฟ"""
        try:
            
            try:
                GPIO.setmode(GPIO.BCM)
            except ValueError:
                pass  
            
            GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)   
            GPIO.output(self.led_pin, GPIO.LOW)   
            
            # Test ไฟ
            GPIO.output(self.led_pin, GPIO.HIGH)  
            time.sleep(0.2)
            GPIO.output(self.led_pin, GPIO.LOW)
            
        except Exception as e:
            print(f"LED setup error: {e}")
    
    def setup_buzzer(self):
        """ตั้งค่า Buzzer (GPIO 17) - โมดูล 3 ขา VCC/GND/I/O, Active-LOW: LOW=ดัง HIGH=เงียบ"""
        try:
            try:
                GPIO.setmode(GPIO.BCM)
            except ValueError:
                pass
            GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.HIGH)  # HIGH=เงียบตอนเริ่ม
        except Exception as e:
            print(f"Buzzer setup error: {e}")
    
    def buzzer_beep(self, duration=0.15):
        """ดัง Buzzer สั้นๆ เมื่อแตะบัตรสำเร็จ (Active-LOW)"""
        try:
            GPIO.output(self.buzzer_pin, GPIO.LOW)   # เปิดเสียง
            time.sleep(duration)
            GPIO.output(self.buzzer_pin, GPIO.HIGH)  # ปิดเสียง
        except Exception as e:
            print(f"Buzzer error: {e}")
    
    def setup_lcd(self):
        """ตั้งค่า LCD I2C (PCF8574 0x27) - แสดงสถานะแบบ 2 บรรทัด"""
        if LCD_AVAILABLE:
            try:
                self.lcd = CharLCD('PCF8574', 0x27)  # Address 0x27
                self.lcd.clear()
                self.lcd.write_string("Insert Item")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("Scan Card")
                print("LCD Display initialized")
                time.sleep(2)
            except Exception as e:
                print(f"LCD setup error: {e}")
                self.lcd = None
        else:
            print("LCD disabled - RPLCD not installed")
    
    # ---------- ส่วน LED Control ----------
    def led_on(self, duration=2):
        """เปิด LED เป็นเวลา duration วินาที"""
        try:
            GPIO.output(self.led_pin, GPIO.HIGH)  # เปิดไฟ
            print("LED ON")
            time.sleep(duration)
            GPIO.output(self.led_pin, GPIO.LOW)   # ปิดไฟ
            print("LED OFF")
        except Exception as e:
            print(f"LED control error: {e}")
    
    def led_blink(self, times=3, interval=0.5):
        """กระพริบ ไฟ"""
        try:
            for i in range(times):
                GPIO.output(self.led_pin, GPIO.HIGH)  # เปิดไฟ
                time.sleep(interval)
                GPIO.output(self.led_pin, GPIO.LOW)   # ปิดไฟ
                time.sleep(interval)
            print(f"LED blinked {times} times")
        except Exception as e:
            print(f"LED blink error: {e}")
    
    def led_off(self):
        """ปิดไฟ"""
        try:
            GPIO.output(self.led_pin, GPIO.LOW)
            print("LED OFF")
        except Exception as e:
            print(f"LED OFF error: {e}")
    
    # ---------- ส่วน LCD Display (ภาษาอังกฤษเพราะ LCD ไม่รองรับไทย) ----------
    def lcd_show_waiting(self):
        """Stage 1: ใส่ขวดก่อน ค่อยแตะบัตร - Insert Item / Scan Card"""
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string("Insert Item")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("Scan Card")
            except Exception as e:
                print(f"LCD display error: {e}")
    
    def lcd_show_rfid(self, card_id):
        """Stage 2: หลังแตะบัตรสำเร็จ - Processing"""
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string("Processing")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("")
            except Exception as e:
                print(f"LCD display error: {e}")
    
    def lcd_show_scanning(self):
        """Stage 3: Processing"""
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string("Processing")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("")
            except Exception as e:
                print(f"LCD display error: {e}")
    
    def lcd_show_results(self, bottle_count, cap_count, label_count, can_count):
        """Stage 4-5: Detected / Bottle / Can"""
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string("Detected")
                self.lcd.cursor_pos = (1, 0)
                if bottle_count > 0 and can_count > 0:
                    self.lcd.write_string("Bottle & Can")
                elif bottle_count > 0:
                    self.lcd.write_string(f"Bottle Cap:{cap_count} L:{label_count}"[:16])
                elif can_count > 0:
                    self.lcd.write_string(f"Can {can_count}")
                else:
                    self.lcd.write_string("Not Detected")
            except Exception as e:
                print(f"LCD display error: {e}")
    
    def lcd_show_score(self, card_id, score):
        """Stage 6: แสดงคนละหน้า - หน้า 1: RFID + Score, หน้า 2: Completed Thank You"""
        if self.lcd:
            try:
                # หน้า 1: RFID + Score
                self.lcd.clear()
                self.lcd.write_string((f"RFID:{card_id}")[:16])
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string((f"Score: {score} pts")[:16])
                time.sleep(3)
                # หน้า 2: Completed Thank You
                self.lcd.clear()
                self.lcd.write_string("Completed")
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string("Thank You")
                time.sleep(3)
            except Exception as e:
                print(f"LCD display error: {e}")

    def signal_handler(self, signum, frame):
        print("\nกำลังหยุด")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    # ---------- ส่วน RFID ----------
    def create_rfid_helper(self):
        """สร้างไฟล์ rfid_helper.py - สคริปแยกสำหรับอ่าน RFID (หลีกเลี่ยง GPIO conflict)"""
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
        """รออ่านบัตร RFID ผ่าน subprocess - return (card_id, text) หรือ (None, None)"""
        print(f"รอสแกน RFID (timeout: {timeout} วินาที)...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout and self.running:
            try:
                # รัน สคริป RFID
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
                            print(f"RFID detected - ID: {card_id}")
                            
                            # เปิด LED เมื่อแตะบัตร RFID
                            GPIO.output(self.led_pin, GPIO.HIGH)
                            print("LED ON")
                            #  ดัง Buzzer เมื่อแตะบัตรสำเร็จ
                            self.buzzer_beep()
                            
                            return card_id, text
                    except json.JSONDecodeError:
                        pass
                
                time.sleep(0.2)
                
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                logger.debug(f"Subprocess error: {e}")
                time.sleep(0.5)
        
        print("RFID timeout")
        return None, None
    
    # ---------- ส่วนกล้อง ----------
    def open_camera(self):
        """เปิดกล้อง USB (0) ขนาด 640x480"""
        try:
            print("เปิดกล้อง...")
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                print("เปิดกล้องไม่ได้")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("เปิดกล้องเเล้ว")
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def capture_image(self):
        """ถ่ายภาพและบันทึกเป็น captured_image.jpg"""
        if not self.camera:
            return None
        
        try:
            print("ถ่ายภาพ...")
            ret, frame = self.camera.read()
            
            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
                print(f"บันทึกภาพ: {image_path}")
                return image_path
            else:
                print("ถ่ายภาพไม่ได้")
                return None
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def close_camera(self):
        """ปิดกล้อง"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("ปิดกล้องแล้ว")
    
    # ---------- ส่วน API ----------
    def send_image_to_api(self, image_path):
        """ส่งภาพไป Cloud AI API (/api/scan) - ได้ผลการตรวจจับ bottle, can, cap, label"""
        try:
            print("ส่งภาพไป AI API")
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = self.session.post(
                    f"{self.api_url}/api/scan",
                    files=files,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                print("ผลจาก API")
                print(f"API Response: {result}")
                return result
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"API request failed: {e}")
            return None
    
    def process_scan_result(self, result_data, card_id=None):
        """คำนวณคะแนน แสดงผล LCD บันทึกไปเว็บ (Bottle +50, Can +100, Cap -10, Label -10)"""
        if not result_data or not result_data.get('success'):
            print("ไม่ได้รับผล")
            return
        
        bottle_count = result_data.get('bottle_count', 0)
        can_count = result_data.get('can_count', 0)
        cap_count = result_data.get('cap_count', 0)
        label_count = result_data.get('label_count', 0)
        
        
        # ขวด PET: +50 คะแนน, กระป๋อง: +100 คะแนน, ฝา: -10 คะแนน, สลาก: -10 คะแนน
        score = (bottle_count * 50) + (can_count * 100) - (cap_count * 10) - (label_count * 10)
        
        print("\nการวิเคราะห์ AI:")
        print("=" * 40)
        print(f"ขวด (Bottles): {bottle_count} (+{bottle_count * 50} คะแนน)")
        print(f"กระป๋อง (Cans): {can_count} (+{can_count * 100} คะแนน)")
        print(f"ฝา (Caps): {cap_count} ({cap_count * -10} คะแนน)")
        print(f"สลาก (Labels): {label_count} ({label_count * -10} คะแนน)")
        print(f"จำนวนรวม: {result_data.get('debug_info', {}).get('total_detections', 0)}")
        print(f"คะแนนรวม: {score}")
        print("=" * 40)
        
        # แสดงผลบน LCD
        self.lcd_show_results(bottle_count, cap_count, label_count, can_count)
        time.sleep(3)
        
        # แสดงคะแนนรวมพร้อมเลขบัตร RFID
        if card_id:
            self.lcd_show_score(card_id, score)
        
        # บันทึกคะแนนไปยังเว็บ
        if card_id:
            self.save_score_to_web(card_id, result_data)
    
    def save_score_to_web(self, card_id, result):
        """ส่งคะแนนไป Member System API (/api/add_score) บันทึกลง database"""
        try:
            
            bottle_count = result.get('bottle_count', 0)
            can_count = result.get('can_count', 0)
            cap_count = result.get('cap_count', 0)
            label_count = result.get('label_count', 0)
            score = (bottle_count * 50) + (can_count * 100) - (cap_count * 10) - (label_count * 10)
            
            
            score_data = {
                'card_id': str(card_id),
                'bottle_count': bottle_count,
                'can_count': can_count,
                'cap_count': cap_count,
                'label_count': label_count,
                'score': score,
                'image_path': 'captured_image.jpg'
            }
            
            print(f"Score Data: {score_data}")
            logger.info(f"Score data to send: {score_data}")
            
            # ส่งไปยัง Member System API
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    web_response = self.session.post(
                        'https://pet-detect-seniorproject-1.onrender.com/api/add_score',  
                        json=score_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if web_response.status_code == 200:
                        print("บันทึกคะแนนสำเร็จ")
                        logger.info(f"Score saved to web: Card {card_id}, Score {result.get('score', 0)}")
                        return
                    else:
                        print(f"ไม่สามารถบันทึกคะแนนได้: {web_response.status_code} (attempt {attempt + 1})")
                        
                except Exception as e:
                    print(f"Web score save error (attempt {attempt + 1}): {e}")
                    logger.error(f"Web score save error (attempt {attempt + 1}): {e}")
                    
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        except Exception as e:
            print(f"Web score save error: {e}")
            logger.debug(f"Web score save failed: {e}")
    
    # ---------- ส่วน Stepper Motor ----------
    def control_stepper(self, result_data):
        """ควบคุม Stepper Motor หมุนคัดแยก - ขวด=ซ้าย, กระป๋อง=ขวา"""
        if not self.stepper:
            print("Stepper motor not available")
            return
        
        if not result_data or not result_data.get('success'):
            return
        
        bottle_count = result_data.get('bottle_count', 0)
        can_count = result_data.get('can_count', 0)
        
        print("\nการควบคุม Stepper Motor:")
        print("=" * 35)
        print(f"Debug - bottle_count: {bottle_count}, can_count: {can_count}")
        print(f"Debug - result_data keys: {list(result_data.keys())}")
        
        try:
            if bottle_count > 0 and can_count > 0:
                print(f"พบทั้งขวด ({bottle_count}) และกระป๋อง ({can_count})")
                print("หมุนขวา")
                self.stepper.move_degrees(90, speed=150)
                time.sleep(2)
                
            elif bottle_count > 0:
                print(f"พบขวด ({bottle_count} อัน)")
                print("หมุนซ้าย")
                self.stepper.move_degrees(-90, speed=150)
                time.sleep(2)
                
            elif can_count > 0:
                print(f"พบกระป๋อง ({can_count} อัน)")
                print("หมุนขวา")
                self.stepper.move_degrees(90, speed=150)
                time.sleep(2)
            else:
                print("ไม่พบขวดหรือกระป๋อง")
                return
            
            # กลับตำแหน่งเดิม
            print("กลับตำแหน่งเดิม")
            self.stepper.return_to_home(speed=200)
            time.sleep(1)
            
        except Exception as e:
            print(f"Stepper control error: {e}")
        
        print("=" * 35)
    
    # ---------- ส่วนหลัก - การสแกน ----------
    def run_single_scan(self):
        """รันการสแกน 1 รอบ: RFID → ถ่ายรูป → ส่ง API → แสดงผล → บันทึกคะแนน → Stepper"""
        print("\nเริ่มการสแกน")
        print("=" * 50)
        
        self.lcd_show_waiting()
        
        # ขั้นตอนที่ 1: สแกน RFID
        card_id, text = self.read_rfid_subprocess(timeout=30)
        if not card_id:
            print("ไม่พบบัตร RFID")
            return False
        
        # ขั้นตอนที่ 2: แสดงเลข RFID บน LCD
        self.lcd_show_rfid(card_id)
        self.lcd_show_scanning()
        
        # ขั้นตอนที่ 3: เปิดกล้องและถ่ายภาพ
        if not self.open_camera():
            return False
        
        image_path = self.capture_image()
        self.close_camera()
        
        if not image_path:
            return False
        
        # ขั้นตอนที่ 4: ส่งภาพไป Cloud AI API
        result = self.send_image_to_api(image_path)
        if not result:
            return False
        
        # ขั้นตอนที่ 5: แสดงผล LCD บันทึกคะแนนไปเว็บ ควบคุม Stepper
        self.process_scan_result(result, card_id)
        self.control_stepper(result)
        
        GPIO.output(self.led_pin, GPIO.LOW)
        print("LED OFF")
        
        print("การสแกนเสร็จสิ้น")
        return True
    
    def run_continuous_scan_system(self):
        """รันระบบสแกนต่อเนื่องแบบ loop - รอ RFID → สแกน → รอ 3 วินาที → วนซ้ำ"""
        print("ระบบสแกน PET ")
        print("=" * 60)
        
        # ตั้งค่า LCD
        self.setup_lcd()
        
        scan_count = 0
        
        while self.running:
            try:
                scan_count += 1
                print(f"\nรอบที่ {scan_count}:")
                
                # ปิดไฟ
                self.led_off()
                
                # แสดงสถานะรอการแตะบัตร
                self.lcd_show_waiting()
                
                success = self.run_single_scan()
                
                if success:
                    print("รอ 3 วินาที")
                    time.sleep(3)
                else:
                    print("รอ 2 วินาที")
                    time.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """ปิดกล้อง LCD LED Buzzer Stepper ลบ rfid_helper.py (ไม่ cleanup Buzzer เพื่อให้เงียบหลังปิด)"""
        try:
            if self.camera:
                self.close_camera()
            if self.stepper:
                self.stepper.cleanup()
            
            # ปิด LCD
            if self.lcd:
                try:
                    self.lcd.clear()
                    self.lcd.write_string("System Shutdown")
                    time.sleep(1)
                    self.lcd.close()
                    print("LCD turned OFF")
                except:
                    pass
            
            # ปิดไฟก่อนจบ
            try:
                GPIO.output(self.led_pin, GPIO.LOW)   # ปิด LED
                print("LED turned OFF")
            except:
                pass
            
            # ลบ สคริป RFID
            if os.path.exists('rfid_helper.py'):
                os.remove('rfid_helper.py')
            
            # ปิด Buzzer (HIGH = เงียบ สำหรับ Active-LOW module)
            try:
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
            except Exception:
                pass
            # Cleanup GPIO เฉพาะ LED, Stepper - ไม่ cleanup ขา Buzzer เพื่อให้คง HIGH (เงียบ)
            try:
                GPIO.cleanup(self.led_pin)
                if self.stepper:
                    GPIO.cleanup(self.stepper.step_pin)
                    GPIO.cleanup(self.stepper.dir_pin)
            except Exception:
                pass
            
            print("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# ============ Main ============
def main():
    """จุดเริ่มต้น - สร้าง client และรันระบบสแกนต่อเนื่อง"""
    print("PET Detect System")
    print("=" * 60)
    
    client = PETDetectSubprocess(api_url=API_URL)
    
    print("System พร้อม")
    
    client.run_continuous_scan_system()
    client.cleanup()

if __name__ == "__main__":
    main()