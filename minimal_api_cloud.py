#!/usr/bin/env python3
"""
Minimal PET Detect API - Cloud Version
สำหรับ Deploy ขึ้น Cloud (Render, Railway, etc.)
Custom YOLOv11n Model: model-yolov11/best.pt
Fallback 1: Standard YOLOv11n (yolo11n.pt) - ultralytics >= 8.3.0
Fallback 2: Standard YOLOv8n (yolov8n.pt)
"""

import logging
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Set YOLO config directory to avoid warning
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

# Suppress YOLO warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
import cv2
import numpy as np
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model with improved fallback handling
try:
    # Fix PyTorch 2.6 weights_only issue
    import torch
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.block.Bottleneck',
        'ultralytics.nn.modules.block.C3',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.head.Detect',
        'ultralytics.nn.modules.block.C2f',
        'ultralytics.nn.modules.block.RepConv'
    ])
    
    # Check if custom model exists first
    custom_model_path = 'model-yolov11/best.pt'
    if os.path.exists(custom_model_path):
        model = YOLO(custom_model_path)
        logger.info("Model loaded successfully from model-yolov11/best.pt (Custom YOLOv11n)")
    else:
        logger.warning(f"Custom model not found at {custom_model_path}, using fallback")
        # ไม่ raise error แต่ให้ไป fallback ต่อ
        model = None
        
    if model:
        logger.info(f"Model classes: {model.names}")
        logger.info(f"Model classes count: {len(model.names)}")
    
except Exception as e:
    logger.error(f"Failed to load custom model: {e}")
    model = None

# Fallback to standard models if custom model failed
if model is None:
    try:
        model = YOLO('yolo11n.pt')
        logger.info("✅ Fallback to standard YOLOv11n model")
        logger.info(f"Model classes: {model.names}")
    except Exception as e2:
        logger.error(f"Failed to load standard YOLOv11n model: {e2}")
        # Final fallback to YOLOv8n
        try:
            model = YOLO('yolov8n.pt')
            logger.info("✅ Final fallback to standard YOLOv8n model")
            logger.info(f"Model classes: {model.names}")
        except Exception as e3:
            logger.error(f"Failed to load YOLOv8n model: {e3}")
            model = None


@app.route('/api/scan', methods=['POST'])
def scan():
    """Image analysis endpoint"""
    if not model:
        return jsonify({
            'success': False,
            'message': 'Model not loaded'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image uploaded'
        }), 400
    
    try:
        # Get image file
        image_file = request.files['image']
        
        # Read image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        logger.info(f"Image processed: {image_array.shape}")
        
        # Run YOLO inference
        results = model(image_array)
        
        # Process results
        bottles = 0
        caps = 0
        labels = 0
        cans = 0
        
        for result in results:
            
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # คำนวณขนาดของ bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # เพิ่ม threshold เป็น 0.6 และตรวจสอบขนาดวัตถุ
                    if confidence > 0.6 and area > 1000:  # วัตถุต้องมีขนาดใหญ่พอ
                        # ตรวจจับโดยไม่ตรวจสอบ aspect ratio
                        if class_name in ["Bottle", "bottle", "ขวด", "water bottle", "wine glass", "cup"]:
                            bottles += 1
                                
                        elif class_name in ["Can", "can", "กระป๋อง", "sports ball", "tennis ball"]:
                            cans += 1
                                
                        elif class_name in ["Cap", "cap", "ฝา", "frisbee", "donut"]:
                            caps += 1
                                
                        elif class_name in ["Label", "label", "ฉลาก", "book", "cell phone", "remote"]:
                            labels += 1
        
        # Calculate score (updated scoring system)
        score = (bottles * 50) + (cans * 100) + (caps * (-10)) + (labels * (-10))
        
        logger.info(f"AI inference completed")
        logger.info(f"Detection results - Bottles: {bottles}, Caps: {caps}, Labels: {labels}, Cans: {cans}, Score: {score}")
        
        return jsonify({
            'success': True,
            'score': score,
            'detections': {
                'bottles': bottles,
                'caps': caps,
                'labels': labels,
                'cans': cans
            },
            # เพิ่ม format ที่ Pi Client คาดหวัง
            'bottle_count': bottles,
            'cap_count': caps,
            'label_count': labels,
            'can_count': cans,
            'debug_info': {
                'confidence_threshold': 0.6,
                'min_area_threshold': 1000,
                'aspect_ratio_validation': False
            },
            'message': f'พบ {bottles + caps + labels + cans} objects: ขวด {bottles}, ฝา {caps}, ฉลาก {labels}, กระป๋อง {cans}'
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting Minimal PET Detect API...")
    print(f"Port: {port}")
    print(f"Model path: model-yolov11/best.pt")
    print(f"Model status: {'Loaded' if model else 'Failed'}")
    print("Available endpoints:")
    print("   - POST /api/scan - Image analysis")
    print("Press Ctrl+C to stop")
    
    # Use gunicorn for production (don't run Flask directly on cloud)
    app.run(host='0.0.0.0', port=port, debug=False)
