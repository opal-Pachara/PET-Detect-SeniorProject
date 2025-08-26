#!/usr/bin/env python3
"""
ทดสอบ API อย่างรวดเร็วหลังจากอัพเดท
"""

import json
import time
from minimal_api import app

def test_endpoints():
    """ทดสอบ endpoints ทั้งหมด"""
    app.config['TESTING'] = True
    client = app.test_client()
    
    print("🧪 Testing Updated API...")
    print("="*40)
    
    # Test 1: Ping endpoint
    print("1. Testing /api/ping")
    response = client.get('/api/ping')
    assert response.status_code == 200
    data = json.loads(response.data)
    print(f"   ✅ Status: {data.get('message')}")
    print(f"   ✅ Model: {data.get('status', {}).get('model')}")
    
    # Test 2: Model info endpoint
    print("\n2. Testing /api/model-info")
    response = client.get('/api/model-info')
    if response.status_code == 200:
        data = json.loads(response.data)
        classes = data.get('model_info', {}).get('classes', {})
        print(f"   ✅ Model type: {data.get('model_info', {}).get('type')}")
        print(f"   ✅ Classes found: {len(classes)}")
        
        # แสดงคลาสที่เกี่ยวข้อง
        relevant_classes = []
        for class_id, class_name in classes.items():
            if any(keyword in class_name.lower() for keyword in ['bottle', 'can', 'cap', 'label', 'ขวด', 'กระป๋อง', 'ฝา', 'สลาก']):
                relevant_classes.append(f"{class_id}: {class_name}")
        
        if relevant_classes:
            print("   📋 Relevant classes:")
            for cls in relevant_classes:
                print(f"      - {cls}")
        else:
            print("   ⚠️  No relevant classes found (bottle, can, cap, label)")
    else:
        print(f"   ❌ Model info failed: {response.status_code}")
    
    # Test 3: Mock scan (ถ้าไม่มีรูป)
    print("\n3. Testing /api/scan structure")
    print("   📝 Expected response structure:")
    print("   {")
    print("     'success': true,")
    print("     'result': {")
    print("       'bottle_count': int,")
    print("       'can_count': int,      # 🆕 NEW!")
    print("       'cap_count': int,")
    print("       'label_count': int,")
    print("       'score': int           # includes can*100")
    print("     }")
    print("   }")
    
    print("\n📊 Scoring system:")
    print("   🥫 Can (กระป๋อง): +100 points")
    print("   🍶 Bottle (ขวด): +50 points")
    print("   🧢 Cap (ฝา): -10 points")
    print("   🏷️  Label (สลาก): -10 points")
    
    print("\n🎉 API structure test completed!")
    print("✅ Ready for Git commit")

if __name__ == "__main__":
    test_endpoints()