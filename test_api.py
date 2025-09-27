#!/usr/bin/env python3
"""
ทดสอบ AI API บน Cloud
"""

import requests
import json
from PIL import Image
import io
import numpy as np

# API URLs
AI_API_URL = "https://pet-detect-ai-api.onrender.com"
WEB_API_URL = "https://pet-detect-seniorproject-1.onrender.com"

def test_ping():
    """ทดสอบ Health Check"""
    print("🔍 ทดสอบ Health Check...")
    try:
        response = requests.get(f"{AI_API_URL}/api/ping", timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """ทดสอบ Root Endpoint"""
    print("\n🔍 ทดสอบ Root Endpoint...")
    try:
        response = requests.get(f"{AI_API_URL}/", timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """ทดสอบ Model Info"""
    print("\n🔍 ทดสอบ Model Info...")
    try:
        response = requests.get(f"{AI_API_URL}/api/model-info", timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_image():
    """สร้างรูปภาพทดสอบ"""
    print("\n🖼️ สร้างรูปภาพทดสอบ...")
    # สร้างรูปภาพสีขาว 640x480
    image = Image.new('RGB', (640, 480), color='white')
    
    # บันทึกรูปภาพ
    image.save('test_image.jpg')
    print("✅ สร้างรูปภาพทดสอบสำเร็จ: test_image.jpg")
    return 'test_image.jpg'

def test_scan():
    """ทดสอบ Image Analysis"""
    print("\n🔍 ทดสอบ Image Analysis...")
    
    # สร้างรูปภาพทดสอบ
    image_path = create_test_image()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{AI_API_URL}/api/scan", files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_web_api():
    """ทดสอบ Web Score System"""
    print("\n🔍 ทดสอบ Web Score System...")
    try:
        response = requests.get(f"{WEB_API_URL}/", timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Length: {len(response.text)} characters")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """ทดสอบทั้งหมด"""
    print("🚀 เริ่มทดสอบ AI API บน Cloud")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_ping),
        ("Root Endpoint", test_root),
        ("Model Info", test_model_info),
        ("Image Analysis", test_scan),
        ("Web Score System", test_web_api)
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
    print("📊 สรุปผลการทดสอบ:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 ผลลัพธ์: {passed}/{len(results)} ผ่าน")
    
    if passed == len(results):
        print("🎉 ทดสอบทั้งหมดผ่าน! API พร้อมใช้งาน")
    else:
        print("⚠️ มีบางส่วนที่ยังไม่ผ่าน ต้องตรวจสอบ")

if __name__ == "__main__":
    main()
