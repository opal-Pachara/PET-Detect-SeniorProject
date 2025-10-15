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
        # SW2=OFF, SW3=OFF, SW6=OFF → ประมาณ 800-1600 steps/rev
        self.steps_per_revolution = 1600  # ปรับตาม DIP switch ที่ตั้งจริง (SW2,3,6=OFF)
        self.current_position = 0  # ตำแหน่งปัจจุบัน (steps)
        
        # Setup GPIO
        self.setup_gpio()
        if self.enable_pin is not None:
            self.enable_motor()
        
        logger.info(f"Professional Stepper Motor Controller initialized")
        logger.info(f"Steps/Rev: {self.steps_per_revolution} (set by DIP switches)")
        logger.info("Use external power supply (12V/24V) for VCC on driver")
        logger.info("Pi provides only signal (PUL+, DIR+) and signal ground")
    
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
    
    
    def cleanup(self):
        """ปิดการใช้งาน stepper motor"""
        try:
            if self.enable_pin is not None:
                GPIO.output(self.enable_pin, GPIO.HIGH)  # HIGH = disable
                logger.info("Motor disabled")
            logger.info("Stepper motor controller cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
