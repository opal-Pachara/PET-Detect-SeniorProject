#!/usr/bin/env python3
"""
PET Detection System - Launcher Script
รันแอปพลิเคชัน PET Detection System ด้วย Streamlit
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    code_dir = current_dir / "code"
    main_file = code_dir / "main.py"
    
    # Check if main.py exists
    if not main_file.exists():
        print("❌ Error: main.py not found in code/ directory")
        print(f"Expected path: {main_file}")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import torch
        import cv2
        import PIL
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if model file exists
    model_file = current_dir / "model-yolov5s" / "best.pt"
    if not model_file.exists():
        print("⚠️  Warning: best.pt model file not found")
        print(f"Expected path: {model_file}")
        print("The application may not work properly without the model file")
    
    # Change to code directory and run streamlit
    os.chdir(code_dir)
    
    print("🚀 Starting PET Detection System...")
    print("📱 Opening web browser...")
    print("🔗 Local URL: http://localhost:8501")
    print("🌐 Network URL: http://192.168.1.x:8501")
    print("\nPress Ctrl+C to stop the application")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 