#!/usr/bin/env python3
"""
ทดสอบ Model ว่าใช้ได้หรือไม่
"""

import requests
import json
from PIL import Image, ImageDraw
import io

# AI API URL
AI_API_URL = "https://pet-detect-ai-api.onrender.com"

def create_test_image():
    """สร้างรูปภาพทดสอบ"""
    print("🖼️ สร้างรูปภาพทดสอบ...")
    
    # สร้างรูปภาพสีขาว 640x480
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # วาดขวด PET ใหญ่ๆ สีเขียวเข้ม
    bottle_x, bottle_y = 250, 100
    bottle_width, bottle_height = 100, 200
    draw.rectangle([bottle_x, bottle_y, bottle_x + bottle_width, bottle_y + bottle_height], 
                   fill='darkgreen', outline='black', width=3)
    
    # วาดฝาขวด (วงกลมสีเขียวเข้ม)
    cap_x, cap_y = bottle_x + bottle_width//2 - 20, bottle_y - 30
    draw.ellipse([cap_x, cap_y, cap_x + 40, cap_y + 40], 
                 fill='green', outline='black', width=3)
    
    # วาดฉลาก (สี่เหลี่ยมสีเหลือง)
    label_x, label_y = bottle_x + 15, bottle_y + 50
    label_width, label_height = 70, 80
    draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], 
                   fill='yellow', outline='orange', width=2)
    
    # บันทึกรูปภาพ
    filename = 'test_model_image.jpg'
    image.save(filename, 'JPEG', quality=95)
    
    print(f"✅ สร้างรูปภาพทดสอบสำเร็จ: {filename}")
    return filename

def test_model_ping():
    """ทดสอบ Model Health Check"""
    print("\n🔍 ทดสอบ Model Health Check...")
    try:
        response = requests.get(f"{AI_API_URL}/api/ping", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_model_info():
    """ทดสอบ Model Info"""
    print("\n🔍 ทดสอบ Model Info...")
    try:
        response = requests.get(f"{AI_API_URL}/api/model-info", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_model_detection():
    """ทดสอบ Model Detection"""
    print("\n🔍 ทดสอบ Model Detection...")
    
    # สร้างรูปภาพทดสอบ
    image_path = create_test_image()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{AI_API_URL}/api/scan", files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # ตรวจสอบ detections
            detections = data.get('detections', {})
            total_objects = sum(detections.values())
            
            print(f"\n📊 Detection Results:")
            print(f"  ขวด (bottles): {detections.get('bottles', 0)}")
            print(f"  ฝา (caps): {detections.get('caps', 0)}")
            print(f"  ฉลาก (labels): {detections.get('labels', 0)}")
            print(f"  กระป๋อง (cans): {detections.get('cans', 0)}")
            print(f"  รวม: {total_objects} objects")
            print(f"  คะแนน: {data.get('score', 0)}")
            
            # ตรวจสอบ debug info
            debug_info = data.get('debug_info', {})
            print(f"\n🔍 Debug Info:")
            print(f"  Model Classes: {debug_info.get('model_classes', [])}")
            print(f"  Total Detections: {debug_info.get('total_detections', 0)}")
            print(f"  Confidence Threshold: {debug_info.get('confidence_threshold', 0)}")
            print(f"  Image Shape: {debug_info.get('image_shape', 'N/A')}")
            
            return total_objects > 0
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def main():
    """ทดสอบ Model ทั้งหมด"""
    print("🤖 เริ่มทดสอบ Model")
    print("=" * 50)
    
    tests = [
        ("Model Health Check", test_model_ping),
        ("Model Info", test_model_info),
        ("Model Detection", test_model_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 สรุปผลการทดสอบ Model:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 ผลลัพธ์: {passed}/{len(results)} ผ่าน")
    
    if passed == len(results):
        print("🎉 Model ทำงานได้ปกติ! พร้อมใช้งาน")
    elif passed >= 2:
        print("⚠️ Model โหลดได้ แต่ Detection อาจมีปัญหา")
    else:
        print("❌ Model มีปัญหา ต้องตรวจสอบ")

if __name__ == "__main__":
    main()
