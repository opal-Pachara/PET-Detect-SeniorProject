#!/usr/bin/env python3
"""
ทดสอบ AI API เท่านั้น
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
    
    # วาดขวด PET (สี่เหลี่ยมสีเขียว)
    bottle_x, bottle_y = 200, 150
    bottle_width, bottle_height = 80, 120
    draw.rectangle([bottle_x, bottle_y, bottle_x + bottle_width, bottle_y + bottle_height], 
                   fill='green', outline='darkgreen', width=2)
    
    # วาดฝาขวด (วงกลมสีเขียวเข้ม)
    cap_x, cap_y = bottle_x + bottle_width//2 - 15, bottle_y - 20
    draw.ellipse([cap_x, cap_y, cap_x + 30, cap_y + 30], 
                 fill='darkgreen', outline='black', width=2)
    
    # วาดฉลาก (สี่เหลี่ยมสีเหลือง)
    label_x, label_y = bottle_x + 10, bottle_y + 30
    label_width, label_height = 60, 40
    draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], 
                   fill='yellow', outline='orange', width=2)
    
    # บันทึกรูปภาพ
    filename = 'test_ai_image.jpg'
    image.save(filename, 'JPEG', quality=95)
    
    print(f"✅ สร้างรูปภาพทดสอบสำเร็จ: {filename}")
    return filename

def test_ai_ping():
    """ทดสอบ AI API Health Check"""
    print("\n🔍 ทดสอบ AI API Health Check...")
    try:
        response = requests.get(f"{AI_API_URL}/api/ping", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_ai_root():
    """ทดสอบ AI API Root"""
    print("\n🔍 ทดสอบ AI API Root...")
    try:
        response = requests.get(f"{AI_API_URL}/", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_ai_model_info():
    """ทดสอบ AI API Model Info"""
    print("\n🔍 ทดสอบ AI API Model Info...")
    try:
        response = requests.get(f"{AI_API_URL}/api/model-info", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_ai_scan():
    """ทดสอบ AI API Image Analysis"""
    print("\n🔍 ทดสอบ AI API Image Analysis...")
    
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
            
            # ตรวจสอบ fields ที่สำคัญ
            if 'success' in data:
                print(f"✅ Success: {data['success']}")
            if 'detections' in data:
                print(f"✅ Detections: {len(data['detections'])} objects")
            if 'score' in data:
                print(f"✅ Score: {data['score']}")
                
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def main():
    """ทดสอบ AI API ทั้งหมด"""
    print("🤖 เริ่มทดสอบ AI API เท่านั้น")
    print("=" * 50)
    
    tests = [
        ("AI Health Check", test_ai_ping),
        ("AI Root Endpoint", test_ai_root),
        ("AI Model Info", test_ai_model_info),
        ("AI Image Analysis", test_ai_scan)
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
    print("📊 สรุปผลการทดสอบ AI API:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 ผลลัพธ์: {passed}/{len(results)} ผ่าน")
    
    if passed == len(results):
        print("🎉 AI API ทดสอบทั้งหมดผ่าน! พร้อมใช้งาน")
    else:
        print("⚠️ มีบางส่วนที่ยังไม่ผ่าน ต้องตรวจสอบ")

if __name__ == "__main__":
    main()
