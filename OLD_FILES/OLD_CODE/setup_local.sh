#!/bin/bash

# PET Detect - Local Setup Script
# สำหรับ Linux/macOS/WSL

echo "🏠 PET Detect Local Setup"
echo "=========================="

# ตรวจสอบ Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# สร้าง virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# เปิดใช้งาน virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# อัพเกรด pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# ติดตั้ง dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# ตรวจสอบโมเดล
if [ ! -f "model-yolov5s/best.pt" ]; then
    echo "❌ Model file not found: model-yolov5s/best.pt"
    echo "📥 Please place your model file in the correct location"
    exit 1
fi

echo "✅ Model file found"

# สร้างโฟลเดอร์ logs
mkdir -p logs

echo ""
echo "🎉 Setup completed successfully!"
echo "📋 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run development server: python run_local.py"
echo "   3. Or run production server: python run_production.py"
echo ""
echo "🌐 API will be available at: http://localhost:5000"