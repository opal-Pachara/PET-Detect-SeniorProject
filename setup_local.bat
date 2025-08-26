@echo off
rem PET Detect - Local Setup Script for Windows

echo 🏠 PET Detect Local Setup
echo ==========================

rem ตรวจสอบ Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

rem สร้าง virtual environment
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

rem เปิดใช้งาน virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

rem อัพเกรด pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

rem ติดตั้ง dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

rem ตรวจสอบโมเดล
if not exist "model-yolov5s\best.pt" (
    echo ❌ Model file not found: model-yolov5s\best.pt
    echo 📥 Please place your model file in the correct location
    pause
    exit /b 1
)

echo ✅ Model file found

rem สร้างโฟลเดอร์ logs
if not exist "logs" mkdir logs

echo.
echo 🎉 Setup completed successfully!
echo 📋 Next steps:
echo    1. Activate virtual environment: venv\Scripts\activate.bat
echo    2. Run development server: python run_local.py
echo    3. Or run production server: python run_production.py
echo.
echo 🌐 API will be available at: http://localhost:5000
pause