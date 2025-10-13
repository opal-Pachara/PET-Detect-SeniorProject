#!/usr/bin/env python3
"""
รัน Minimal API สำหรับโมเดล AI เท่านั้น
ไม่มี web interface, database, authentication
"""

import os
import sys
import subprocess

def install_minimal_requirements():
    """ติดตั้ง dependencies แบบ minimal"""
    print("🔧 Installing minimal requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_minimal.txt"
        ])
        print("✅ Minimal dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements_minimal.txt not found")
        return False
    return True

def check_model_file():
    """ตรวจสอบไฟล์โมเดล"""
    model_path = "model-yolov5s/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("📥 Please ensure your model file is in the correct location")
        print("   You can download or copy your trained model to this path")
        return False
    
    # ตรวจสอบขนาดไฟล์
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Model file found: {model_path} ({file_size:.1f} MB)")
    return True

def run_minimal_api():
    """รัน Minimal API"""
    print("🚀 Starting Minimal PET Detect API...")
    print("🎯 Features included:")
    print("   - AI Model inference")
    print("   - Image analysis")
    print("   - Basic health check")
    print("🚫 Features excluded:")
    print("   - Web interface")
    print("   - User authentication")
    print("   - Database storage")
    print("\n🌐 Server will be available at: http://localhost:5000")
    print("📡 API endpoints:")
    print("   - POST /api/scan - AI image analysis")
    print("   - GET /api/ping - Health check")
    print("   - GET /api/model-info - Model information")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Import และรัน minimal API
        from minimal_api import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except ImportError as e:
        print(f"❌ Error importing minimal API: {e}")
        print("📁 Make sure minimal_api.py exists in the current directory")
    except Exception as e:
        print(f"❌ Error running API: {e}")

def main():
    print("🎯 PET Detect - Minimal API Setup")
    print("=" * 50)
    print("📦 This will install only essential packages for:")
    print("   - Flask API server")
    print("   - AI model inference") 
    print("   - Basic image processing")
    print("=" * 50)
    
    # ตรวจสอบ Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    os.makedirs("model-yolov5s", exist_ok=True)
    
    # ติดตั้ง dependencies
    if not install_minimal_requirements():
        return
    
    # ตรวจสอบโมเดล
    if not check_model_file():
        print("\n⚠️  Warning: Model file not found!")
        print("   The API will start but image analysis will fail.")
        response = input("   Continue anyway? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    
    print("\n" + "=" * 50)
    
    # รัน API
    run_minimal_api()

if __name__ == "__main__":
    main()