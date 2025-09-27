#!/usr/bin/env python3
"""
Minimal PET Detect API - Cloud Version
สำหรับ Deploy ขึ้น Cloud (Render, Railway, etc.)
Custom YOLOv11n Model: model-yolov5s/best.pt
Fallback: Standard YOLOv11n (yolov11n.pt)
"""

import logging
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model
try:
    # Fix PyTorch 2.6 weights_only issue
    import torch
    torch.serialization.add_safe_globals([
        'ultralytics.nn.tasks.DetectionModel',
        'ultralytics.nn.modules.block.Bottleneck',
        'ultralytics.nn.modules.block.C3',
        'ultralytics.nn.modules.block.SPPF',
        'ultralytics.nn.modules.conv.Conv',
        'ultralytics.nn.modules.head.Detect'
    ])
    
    # Try loading custom model first (YOLOv11n)
    model = YOLO('model-yolov5s/best.pt')
    logger.info("Model loaded successfully from model-yolov5s/best.pt (YOLOv11n)")
except Exception as e:
    logger.error(f"Failed to load custom model: {e}")
    # Fallback to standard YOLOv11n
    try:
        model = YOLO('yolov11n.pt')
        logger.info("Fallback to standard YOLOv11n model")
    except Exception as e2:
        logger.error(f"Failed to load standard YOLOv11n model: {e2}")
        model = None

@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'PET Detect API is running',
        'model_loaded': model is not None
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_path': 'model-yolov5s/best.pt',
        'model_loaded': model is not None,
        'model_status': 'Loaded' if model else 'Failed'
    })

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
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                if confidence > 0.5:  # Confidence threshold
                    if class_name in ["bottle", "ขวด"]:
                        bottles += 1
                    elif class_name in ["cap", "ฝา"]:
                        caps += 1
                    elif class_name in ["label", "ฉลาก"]:
                        labels += 1
                    elif class_name in ["can", "กระป๋อง"]:
                        cans += 1
        
        # Calculate score
        score = (bottles * 50) + (caps * 10) + (labels * 5) + (cans * 30)
        
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
            }
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
    print(f"Model path: model-yolov5s/best.pt")
    print(f"Model status: {'Loaded' if model else 'Failed'}")
    print("Available endpoints:")
    print("   - POST /api/scan - Image analysis")
    print("   - GET /api/ping - Health check")
    print("   - GET /api/model-info - Model information")
    print("Press Ctrl+C to stop")
    
    # Use gunicorn for production
    app.run(host='0.0.0.0', port=port, debug=False)
