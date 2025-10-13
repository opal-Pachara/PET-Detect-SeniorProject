#!/usr/bin/env python3
"""
Test USB Camera for Raspberry Pi
ทดสอบการทำงานของ USB Camera
"""

import cv2
import time

def test_usb_camera():
    """Test USB camera functionality"""
    print("🔍 ทดสอบ USB Camera...")
    
    # Try different camera indices
    for camera_index in [0, 1]:
        print(f"📷 ทดสอบ Camera Index: {camera_index}")
        
        # Initialize camera
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            print(f"❌ ไม่สามารถเปิด Camera Index {camera_index} ได้")
            continue
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ เปิด Camera Index {camera_index} สำเร็จ")
        
        # Capture test image
        print("📸 กำลังถ่ายภาพทดสอบ...")
        ret, frame = camera.read()
        
        if ret:
            # Save test image
            timestamp = int(time.time())
            image_path = f"test_camera_{camera_index}_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"✅ ถ่ายภาพทดสอบสำเร็จ: {image_path}")
            
            # Get camera info
            width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = camera.get(cv2.CAP_PROP_FPS)
            
            print(f"📊 ข้อมูล Camera:")
            print(f"   - ความกว้าง: {width}")
            print(f"   - ความสูง: {height}")
            print(f"   - FPS: {fps}")
            
            # Release camera
            camera.release()
            cv2.destroyAllWindows()
            
            print(f"✅ Camera Index {camera_index} ทำงานปกติ")
            return camera_index
            
        else:
            print(f"❌ ไม่สามารถถ่ายภาพจาก Camera Index {camera_index} ได้")
            camera.release()
            continue
    
    print("❌ ไม่พบ USB Camera ที่ใช้งานได้")
    return None

def list_camera_devices():
    """List available camera devices"""
    print("📋 รายการ Camera Devices:")
    
    # Check for v4l2-ctl command
    import subprocess
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ ไม่สามารถรัน v4l2-ctl ได้")
    except FileNotFoundError:
        print("❌ ไม่พบ v4l2-ctl command")
        print("💡 ติดตั้ง: sudo apt-get install v4l-utils")

def main():
    """Main function"""
    print("🎯 ทดสอบ USB Camera สำหรับ Raspberry Pi")
    print("=" * 50)
    
    # List camera devices
    list_camera_devices()
    print()
    
    # Test USB camera
    working_camera = test_usb_camera()
    
    if working_camera is not None:
        print(f"\n✅ พบ USB Camera ที่ใช้งานได้: Index {working_camera}")
        print("🚀 ระบบพร้อมใช้งาน!")
    else:
        print("\n❌ ไม่พบ USB Camera ที่ใช้งานได้")
        print("💡 ตรวจสอบ:")
        print("   1. การเชื่อมต่อ USB Camera")
        print("   2. รัน 'lsusb' เพื่อดู USB devices")
        print("   3. รัน 'v4l2-ctl --list-devices' เพื่อดู camera devices")

if __name__ == "__main__":
    main() 