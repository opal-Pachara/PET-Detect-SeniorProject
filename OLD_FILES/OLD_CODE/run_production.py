#!/usr/bin/env python3
"""
รันแบบ Production-like บนเครื่องตัวเอง
ใช้ Gunicorn WSGI server
"""

import os
import sys
import subprocess

def install_production_requirements():
    """ติดตั้ง dependencies สำหรับ production"""
    print("🔧 Installing production packages...")
    
    # เพิ่ม gunicorn ลงใน requirements
    production_packages = [
        "gunicorn>=20.1.0",
        "gevent>=22.10.0"  # สำหรับ async workers
    ]
    
    try:
        # ติดตั้ง requirements.txt ก่อน
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # ติดตั้งแพ็คเกจเพิ่มเติม
        for package in production_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
        print("✅ Production dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def create_gunicorn_config():
    """สร้างไฟล์ config สำหรับ Gunicorn"""
    config_content = """# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2  # จำนวน worker processes
worker_class = "gevent"  # ใช้ gevent สำหรับ async
worker_connections = 1000
timeout = 120  # เพิ่ม timeout สำหรับ ML inference
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True  # โหลดแอปก่อนสร้าง workers

# Logging
accesslog = "access.log"
errorlog = "error.log"
loglevel = "info"

# Process naming
proc_name = "pet_detect_api"

def when_ready(server):
    print("🚀 PET Detect API Server is ready!")
    print(f"🌐 Server running on: http://localhost:5000")
    print("📡 API endpoints available:")
    print("   - POST /api/scan - Image analysis")
    print("   - POST /api/register - User registration")
    print("   - POST /api/login - User login")
"""
    
    with open("gunicorn_config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Gunicorn config created")

def run_gunicorn():
    """รัน Gunicorn server"""
    print("🚀 Starting Gunicorn server...")
    
    try:
        cmd = [
            "gunicorn",
            "--config", "gunicorn_config.py",
            "code.api:app"
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except FileNotFoundError:
        print("❌ Gunicorn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn"])
        print("✅ Please run again")

def main():
    print("🏭 PET Detect - Production Deployment Setup")
    print("=" * 50)
    
    # ติดตั้ง dependencies
    if not install_production_requirements():
        return
    
    # สร้าง config
    create_gunicorn_config()
    
    # ตรวจสอบโมเดล
    model_path = "model-yolov5s/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("\n" + "=" * 50)
    
    # รัน Gunicorn
    run_gunicorn()

if __name__ == "__main__":
    main()