"""
Stepper Motor Controller with Micro Stepping
สำหรับควบคุม Stepper Motor แบบ Micro Step ด้วย Raspberry Pi
รองรับ A4988, DRV8825, TMC2208 Driver
"""

import RPi.GPIO as GPIO
import time
import logging

logger = logging.getLogger(__name__)

class StepperMotorController:
    """ควบคุม Stepper Motor แบบ Micro Step"""
    
    def __init__(self, step_pin=20, dir_pin=21, enable_pin=None):
        """
        Initialize Stepper Motor Controller for Professional Driver
        
        Args:
            step_pin: GPIO pin สำหรับ PUL+ (STEP signal)
            dir_pin: GPIO pin สำหรับ DIR+ (direction)
            enable_pin: GPIO pin สำหรับ ENA+ (optional, None = ไม่ใช้)
        """
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        
        # Professional Driver Settings
        # Micro step จะตั้งค่าผ่าน DIP switches บน driver แล้ว
        self.steps_per_revolution = 3200  # สำหรับ 1/16 micro step (ปรับตาม DIP switch)
        self.current_position = 0  # ตำแหน่งปัจจุบัน (steps)
        
        # Setup GPIO
        self.setup_gpio()
        if self.enable_pin is not None:
            self.enable_motor()
        
        logger.info(f"Professional Stepper Motor Controller initialized")
        logger.info(f"Steps/Rev: {self.steps_per_revolution} (set by DIP switches)")
        logger.info("⚡ Use external power supply (12V/24V) for VCC on driver")
        logger.info("🔌 Pi provides only signal (PUL+, DIR+) and signal ground")
    
    def setup_gpio(self):
        """ตั้งค่า GPIO pins"""
        try:
            GPIO.setmode(GPIO.BCM)
        except ValueError:
            # GPIO mode already set, just continue
            logger.info("GPIO mode already set, continuing...")
        
        # Output pins สำหรับ Professional Driver
        pins = [self.step_pin, self.dir_pin]
        if self.enable_pin is not None:
            pins.append(self.enable_pin)
        
        for pin in pins:
            try:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            except RuntimeError as e:
                logger.warning(f"GPIO pin {pin} setup warning: {e}")
        
        enable_status = "with Enable pin" if self.enable_pin else "without Enable pin"
        logger.info(f"GPIO pins configured for professional driver ({enable_status})")
    
    def set_steps_per_revolution(self, steps):
        """ตั้งค่า steps ต่อรอบ (ตาม DIP switch configuration)"""
        self.steps_per_revolution = steps
        logger.info(f"Steps per revolution set to: {steps}")
        
        # แสดงตาราง micro step modes ทั่วไป
        print("Micro Step Modes (ตั้งค่าผ่าน DIP switches):")
        print("FULL STEP:    200 steps/rev")
        print("1/2 STEP:     400 steps/rev") 
        print("1/4 STEP:     800 steps/rev")
        print("1/8 STEP:     1600 steps/rev")
        print("1/16 STEP:    3200 steps/rev")
        print("1/32 STEP:    6400 steps/rev")
        print(f"Current:      {steps} steps/rev")
    
    def enable_motor(self):
        """เปิดใช้งานมอเตอร์"""
        if self.enable_pin is not None:
            GPIO.output(self.enable_pin, GPIO.LOW)  # LOW = enable
            logger.info("Motor enabled")
        else:
            logger.info("Motor always enabled (no enable pin)")
    
    def disable_motor(self):
        """ปิดการใช้งานมอเตอร์"""
        if self.enable_pin is not None:
            GPIO.output(self.enable_pin, GPIO.HIGH)  # HIGH = disable
            logger.info("Motor disabled")
        else:
            logger.info("Cannot disable motor (no enable pin)")
    
    def set_direction(self, clockwise=True):
        """ตั้งค่าทิศทางการหมุน"""
        GPIO.output(self.dir_pin, GPIO.HIGH if clockwise else GPIO.LOW)
        direction = "Clockwise" if clockwise else "Counter-clockwise"
        logger.debug(f"Direction set to: {direction}")
    
    def step_once(self, delay=0.001):
        """ขับมอเตอร์ 1 step"""
        GPIO.output(self.step_pin, GPIO.HIGH)
        time.sleep(delay / 2)
        GPIO.output(self.step_pin, GPIO.LOW)
        time.sleep(delay / 2)
    
    def move_steps(self, steps, speed=1000, clockwise=True):
        """
        เคลื่อนที่จำนวน steps ที่กำหนด
        
        Args:
            steps: จำนวน steps ที่ต้องการเคลื่อนที่
            speed: ความเร็ว (steps per second)
            clockwise: ทิศทางการหมุน
        """
        if steps <= 0:
            return
        
        self.set_direction(clockwise)
        delay = 1.0 / speed  # delay ระหว่าง steps
        
        direction_multiplier = 1 if clockwise else -1
        
        logger.info(f"Moving {steps} steps at {speed} steps/sec, direction: {'CW' if clockwise else 'CCW'}")
        
        try:
            for i in range(steps):
                self.step_once(delay)
                self.current_position += direction_multiplier
                
                # แสดงความคืบหน้าทุก 100 steps
                if (i + 1) % 100 == 0:
                    print(f"Progress: {i + 1}/{steps} steps")
        
        except Exception as e:
            logger.error(f"Error during movement: {e}")
        
        logger.info(f"Movement completed. Current position: {self.current_position}")
    
    def move_degrees(self, degrees, speed=1000):
        """
        หมุนตามองศาที่กำหนด
        
        Args:
            degrees: องศาที่ต้องการหมุน (+/- สำหรับทิศทาง)
            speed: ความเร็ว (steps per second)
        """
        steps = abs(int((degrees / 360.0) * self.steps_per_revolution))
        clockwise = degrees > 0
        
        logger.info(f"Moving {degrees} degrees = {steps} steps")
        self.move_steps(steps, speed, clockwise)
    
    def rotate_left(self, degrees=90, speed=800):
        """หมุนซ้าย (สำหรับขวด)"""
        print(f"หมุนซ้าย {degrees} องศา (ขวด)")
        self.move_degrees(-degrees, speed)  # ลบเพื่อหมุนซ้าย
    
    def rotate_right(self, degrees=90, speed=800):
        """หมุนขวา (สำหรับกระป๋อง)"""
        print(f"หมุนขวา {degrees} องศา (กระป๋อง)")
        self.move_degrees(degrees, speed)  # บวกเพื่อหมุนขวา
    
    def return_to_home(self, speed=1200):
        """กลับไปตำแหน่งเริ่มต้น (position 0)"""
        if self.current_position == 0:
            print("อยู่ตำแหน่งเริ่มต้นแล้ว")
            return
        
        steps_to_home = abs(self.current_position)
        clockwise = self.current_position < 0  # ถ้าตำแหน่งติดลบ ให้หมุนขวา
        
        print(f"กลับตำแหน่งเริ่มต้น: {steps_to_home} steps")
        logger.info(f"Returning to home position from {self.current_position}")
        
        self.move_steps(steps_to_home, speed, clockwise)
        self.current_position = 0  # รีเซ็ตตำแหน่ง
    
    def get_position_degrees(self):
        """ดูตำแหน่งปัจจุบันในหน่วยองศา"""
        degrees = (self.current_position / self.steps_per_revolution) * 360
        return round(degrees, 2)
    
    def calibrate_position(self):
        """รีเซ็ตตำแหน่งปัจจุบันเป็น 0 (ใช้เมื่อปรับตำแหน่งมือ)"""
        self.current_position = 0
        logger.info("Position calibrated to 0")
        print("ตำแหน่งปัจจุบันถูกตั้งเป็น 0 องศา")
    
    def cleanup(self):
        """ปิดการใช้งาน stepper motor"""
        try:
            self.disable_motor()
            logger.info("Stepper motor controller cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def test_stepper_motor():
    """ทดสอบ Stepper Motor"""
    print("🔧 ทดสอบ Stepper Motor Controller")
    print("=" * 50)
    
    # สร้าง controller
    stepper = StepperMotorController()
    
    try:
        # แสดงข้อมูลปัจจุบัน
        print(f"📍 โหมดปัจจุบัน: {stepper.current_mode}")
        print(f"📏 Steps/Revolution: {stepper.steps_per_revolution}")
        print(f"📐 ตำแหน่งปัจจุบัน: {stepper.get_position_degrees()} องศา")
        
        # ทดสอบการหมุน
        print("\n🔄 ทดสอบการหมุน...")
        
        # หมุนขวา 90 องศา
        print("1. หมุนขวา 90 องศา")
        stepper.rotate_right(90, speed=1000)
        time.sleep(1)
        
        # หมุนซ้าย 180 องศา
        print("2. หมุนซ้าย 180 องศา")
        stepper.rotate_left(180, speed=1000)
        time.sleep(1)
        
        # กลับตำแหน่งเริ่มต้น
        print("3. กลับตำแหน่งเริ่มต้น")
        stepper.return_to_home(speed=1200)
        
        print(f"📐 ตำแหน่งสุดท้าย: {stepper.get_position_degrees()} องศา")
        print("✅ การทดสอบเสร็จสิ้น!")
        
    except KeyboardInterrupt:
        print("\n🛑 การทดสอบถูกยกเลิก")
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    
    finally:
        stepper.cleanup()
        GPIO.cleanup()

if __name__ == "__main__":
    test_stepper_motor()