#!/usr/bin/env python3
"""
Debug API Response
ตรวจสอบ API response จริงๆ
"""

import requests
import json

def debug_api():
    """Debug API response"""
    print("🔍 ตรวจสอบ API Response...")
    
    # Send test request
    url = "https://pet-detect-seniorproject-production.up.railway.app/api/scan"
    
    # Use a test image (you can replace with actual image)
    with open("usb_image.jpg", "rb") as img_file:
        files = {'image': img_file}
        response = requests.post(url, files=files)
    
    print(f"📡 Status code: {response.status_code}")
    print(f"📄 Response text: {response.text}")
    
    try:
        result = response.json()
        print("📊 JSON Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check structure
        print("\n🔍 ตรวจสอบโครงสร้าง:")
        print(f"result type: {type(result)}")
        if 'result' in result:
            print(f"result['result'] type: {type(result['result'])}")
            print(f"result['result'] keys: {list(result['result'].keys())}")
        else:
            print("❌ ไม่มี 'result' key")
            print(f"Available keys: {list(result.keys())}")
            
    except Exception as e:
        print(f"❌ JSON error: {e}")

if __name__ == "__main__":
    debug_api() 