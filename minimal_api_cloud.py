#!/usr/bin/env python3
"""
Minimal PET Detect API - Cloud Version
สำหรับ Deploy ขึ้น Cloud (Render, Railway, etc.)
Custom YOLOv11n Model: model-yolov5s/best.pt
Fallback 1: Standard YOLOv11n (yolo11n.pt) - ultralytics >= 8.3.0
Fallback 2: Standard YOLOv8n (yolov8n.pt)
"""

import logging
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Set YOLO config directory to avoid warning
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'
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
    # Fix PyTorch weights_only issue
    import torch
    import torch.serialization
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
    
    # Set weights_only=False for older models
    import os
    os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
    
    # Try loading custom model first
    try:
        model = YOLO('model-yolov5s/best.pt')
        logger.info("Model loaded successfully from model-yolov5s/best.pt")
        logger.info(f"Model classes: {model.names}")
        logger.info(f"Model classes count: {len(model.names)}")
    except Exception as e:
        logger.error(f"Failed to load custom model: {e}")
        # Fallback to YOLOv8n (more stable)
        try:
            model = YOLO('yolov8n.pt')
            logger.info("Fallback to standard YOLOv8n model")
        except Exception as e2:
            logger.error(f"Failed to load YOLOv8n model: {e2}")
            model = None
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'PET Detect API is running',
        'model_loaded': model is not None,
        'endpoints': ['/api/ping', '/api/scan', '/api/model-info']
    })

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
    if model:
        return jsonify({
            'model_path': 'model-yolov5s/best.pt',
            'model_loaded': True,
            'model_status': 'Loaded',
            'model_classes': list(model.names.values()),
            'model_classes_dict': dict(model.names),
            'model_classes_count': len(model.names)
        })
    else:
        return jsonify({
            'model_path': 'model-yolov5s/best.pt',
            'model_loaded': False,
            'model_status': 'Failed',
            'error': 'Model not loaded'
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
        
        # Debug: Log model classes
        logger.info(f"Model classes: {model.names}")
        logger.info(f"Model classes list: {list(model.names.values())}")
        
        for result in results:
            logger.info(f"Number of detections: {len(result.boxes) if result.boxes is not None else 0}")
            
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    logger.info(f"Detection {i+1}: class='{class_name}' (id={class_id}), confidence={confidence:.3f}")
                    logger.info(f"  Class name type: {type(class_name)}, value: '{class_name}'")
                    
                    # ลด threshold เป็น 0.1 เพื่อดู detections ทั้งหมด
                    if confidence > 0.1:
                        # แก้ไข class names ให้ตรงกับ model
                        if class_name in ["Bottle", "bottle", "ขวด"]:
                            bottles += 1
                            logger.info(f"✅ Counted bottle: {class_name}")
                        elif class_name in ["Cap", "cap", "ฝา"]:
                            caps += 1
                            logger.info(f"✅ Counted cap: {class_name}")
                        elif class_name in ["Label", "label", "ฉลาก"]:
                            labels += 1
                            logger.info(f"✅ Counted label: {class_name}")
                        elif class_name in ["Can", "can", "กระป๋อง"]:
                            cans += 1
                            logger.info(f"✅ Counted can: {class_name}")
                        else:
                            logger.info(f"❓ Unknown class: {class_name} (confidence: {confidence:.3f})")
                    else:
                        logger.info(f"Low confidence detection: {class_name} ({confidence:.3f})")
        
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
            },
            # เพิ่ม format ที่ Pi Client คาดหวัง
            'bottle_count': bottles,
            'cap_count': caps,
            'label_count': labels,
            'can_count': cans,
            'debug_info': {
                'model_classes': list(model.names.values()),
                'model_classes_dict': dict(model.names),
                'total_detections': len(result.boxes) if result.boxes is not None else 0,
                'confidence_threshold': 0.1,
                'image_shape': image_array.shape,
                'processing_time': 'N/A'
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
    print(f"Model path: model-yolov5s/best.pt")
    print(f"Model status: {'Loaded' if model else 'Failed'}")
    print("Available endpoints:")
    print("   - POST /api/scan - Image analysis")
    print("   - GET /api/ping - Health check")
    print("   - GET /api/model-info - Model information")
    print("Press Ctrl+C to stop")
    
    # Use gunicorn for production (don't run Flask directly on cloud)
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=port, debug=False)
