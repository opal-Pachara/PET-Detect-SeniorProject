#!/usr/bin/env python3
"""
RFID + USB Camera + API System for Raspberry Pi
Combines RFID scanning, USB camera capture, and API communication
"""

import cv2
import requests
import json
import time
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import threading
import os
from datetime import datetime

class RFIDUSBCameraSystem:
    def __init__(self, api_url="https://pet-detect-seniorproject-production.up.railway.app/api/scan", camera_index=0):
        """
        Initialize RFID + USB Camera + API system
        
        Args:
            api_url (str): URL of the cloud API endpoint
            camera_index (int): USB camera index (usually 0)
        """
        self.api_url = api_url
        self.camera_index = camera_index
        self.rfid_reader = SimpleMFRC522()
        
        # Initialize USB camera
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception(f"Could not open camera at index {camera_index}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # GPIO setup for LED indicators
        GPIO.setmode(GPIO.BCM)
        self.led_green = 18  # Green LED for success
        self.led_red = 17     # Red LED for error
        self.led_blue = 27    # Blue LED for scanning
        
        GPIO.setup(self.led_green, GPIO.OUT)
        GPIO.setup(self.led_red, GPIO.OUT)
        GPIO.setup(self.led_blue, GPIO.OUT)
        
        # Turn off all LEDs initially
        self.led_off()
        
        print("🔧 RFID + USB Camera + API System initialized")
        print(f"📡 API URL: {self.api_url}")
        print(f"📷 USB Camera Index: {self.camera_index}")
    
    def led_on(self, color):
        """Turn on specific LED"""
        if color == "green":
            GPIO.output(self.led_green, GPIO.HIGH)
        elif color == "red":
            GPIO.output(self.led_red, GPIO.HIGH)
        elif color == "blue":
            GPIO.output(self.led_blue, GPIO.HIGH)
    
    def led_off(self):
        """Turn off all LEDs"""
        GPIO.output(self.led_green, GPIO.LOW)
        GPIO.output(self.led_red, GPIO.LOW)
        GPIO.output(self.led_blue, GPIO.LOW)
    
    def scan_rfid(self):
        """
        Scan RFID card and return card data
        
        Returns:
            tuple: (card_id, card_text) or (None, None) if no card
        """
        try:
            print("🔍 Waiting for RFID card...")
            self.led_on("blue")  # Blue LED indicates scanning
            
            # Scan for RFID card
            card_id, card_text = self.rfid_reader.read()
            
            self.led_off()
            print(f"✅ RFID Card detected!")
            print(f"   Card ID: {card_id}")
            print(f"   Card Text: {card_text}")
            
            return card_id, card_text
            
        except Exception as e:
            self.led_off()
            print(f"❌ RFID scan error: {e}")
            return None, None
    
    def capture_image(self, filename=None):
        """
        Capture image using USB camera
        
        Args:
            filename (str): Optional filename for the image
            
        Returns:
            str: Path to captured image file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
            
            filepath = f"/home/pi/images/{filename}"
            
            # Ensure directory exists
            os.makedirs("/home/pi/images", exist_ok=True)
            
            print("📸 Capturing image...")
            self.led_on("blue")
            
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret:
                raise Exception("Failed to capture frame from USB camera")
            
            # Save image
            cv2.imwrite(filepath, frame)
            
            self.led_off()
            print(f"✅ Image captured: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.led_off()
            print(f"❌ Image capture error: {e}")
            return None
    
    def send_to_api(self, image_path, rfid_data=None):
        """
        Send image to cloud API
        
        Args:
            image_path (str): Path to image file
            rfid_data (dict): Optional RFID data to include
            
        Returns:
            dict: API response or None if error
        """
        try:
            print("📡 Sending to API...")
            self.led_on("blue")
            
            # Prepare files and data
            files = {'image': open(image_path, 'rb')}
            data = {}
            
            # Add RFID data if provided
            if rfid_data:
                data['rfid_id'] = rfid_data.get('card_id')
                data['rfid_text'] = rfid_data.get('card_text')
                data['timestamp'] = datetime.now().isoformat()
            
            # Send POST request
            response = requests.post(self.api_url, files=files, data=data)
            
            self.led_off()
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API Response received:")
                print(f"   Bottle Count: {result.get('bottle_count', 0)}")
                print(f"   Cap Count: {result.get('cap_count', 0)}")
                print(f"   Label Count: {result.get('label_count', 0)}")
                print(f"   Score: {result.get('score', 0)}")
                
                self.led_on("green")  # Success LED
                time.sleep(2)
                self.led_off()
                
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
                self.led_on("red")  # Error LED
                time.sleep(2)
                self.led_off()
                
                return None
                
        except Exception as e:
            self.led_off()
            print(f"❌ API communication error: {e}")
            return None
    
    def run_loop(self):
        """
        Main loop: Scan RFID → Capture Image → Send to API
        """
        print("\n🚀 Starting RFID + USB Camera + API Loop")
        print("=" * 50)
        
        while True:
            try:
                print("\n" + "="*50)
                print("🔄 Starting new cycle...")
                
                # Step 1: Scan RFID
                card_id, card_text = self.scan_rfid()
                
                if card_id is None:
                    print("⏭️ No RFID card detected, skipping...")
                    time.sleep(2)
                    continue
                
                # Step 2: Capture Image
                image_path = self.capture_image()
                
                if image_path is None:
                    print("❌ Failed to capture image")
                    continue
                
                # Step 3: Send to API
                rfid_data = {
                    'card_id': card_id,
                    'card_text': card_text
                }
                
                api_result = self.send_to_api(image_path, rfid_data)
                
                if api_result:
                    print("🎉 Cycle completed successfully!")
                else:
                    print("⚠️ Cycle completed with errors")
                
                # Wait before next cycle
                print("⏳ Waiting 5 seconds before next cycle...")
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping system...")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                time.sleep(3)
    
    def cleanup(self):
        """Clean up GPIO and camera resources"""
        print("🧹 Cleaning up...")
        self.led_off()
        GPIO.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")

def main():
    """Main function"""
    print("🎯 PET Detection System - RFID + USB Camera + API")
    print("=" * 60)
    
    # Initialize system
    system = RFIDUSBCameraSystem()
    
    try:
        # Run main loop
        system.run_loop()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Cleanup
        system.cleanup()

if __name__ == "__main__":
    main() 