#!/usr/bin/env python3
"""
รันบนเครื่องตัวเอง (Local Development Server)
สำหรับ testing และ development
"""

import os
import sys
import subprocess

def install_requirements():
    """ติดตั้ง dependencies ที่จำเป็น"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def check_model_file():
    """ตรวจสอบไฟล์โมเดล"""
    model_path = "model-yolov5s/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("📥 Please ensure your model file is in the correct location")
        return False
    print(f"✅ Model file found: {model_path}")
    return True

def run_flask_app():
    """รัน Flask application"""
    print("🚀 Starting Flask API server...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📡 API endpoints:")
    print("   - POST /api/scan - สำหรับวิเคราะห์รูปภาพ")
    print("   - POST /api/register - สำหรับสมัครสมาชิก")
    print("   - POST /api/login - สำหรับเข้าสู่ระบบ")
    print("\n⏹️  Press Ctrl+C to stop the server\n")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'code/api.py'
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    try:
        from code.api import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("📁 Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Error running Flask app: {e}")

def main():
    print("🏠 PET Detect - Local Deployment Setup")
    print("=" * 50)
    
    # ตรวจสอบ Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # ติดตั้ง dependencies
    if not install_requirements():
        return
    
    # ตรวจสอบโมเดล
    if not check_model_file():
        return
    
    print("\n" + "=" * 50)
    
    # รัน Flask app
    run_flask_app()

if __name__ == "__main__":
    main()