"""
Minimal API สำหรับโมเดล AI และ Raspberry Pi
เฉพาะ endpoints ที่จำเป็น: /api/scan และ /api/ping
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load YOLO model
MODEL_PATH = 'model-yolov5s/best.pt'
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        model = None
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

@app.route('/api/scan', methods=['POST'])
def scan():
    """
    วิเคราะห์รูปภาพด้วยโมเดล AI
    Input: image file (multipart/form-data)
    Output: จำนวน bottle, cap, label และคะแนน
    """
    try:
        if model is None:
            return jsonify({
                'success': False, 
                'message': 'Model not loaded',
                'error': 'AI_MODEL_ERROR'
            }), 500
            
        if 'image' not in request.files:
            return jsonify({
                'success': False, 
                'message': 'No image uploaded',
                'error': 'NO_IMAGE'
            }), 400
            
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False, 
                'message': 'Empty image file',
                'error': 'EMPTY_FILE'
            }), 400

        # Convert image to numpy array
        try:
            image = Image.open(image_file.stream).convert("RGB")
            image_np = np.array(image)
            logger.info(f"📷 Image processed: {image.size}")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Invalid image format: {str(e)}',
                'error': 'INVALID_IMAGE'
            }), 400

        # Run AI detection
        try:
            results = model(image_np)
            result = results[0]
            logger.info(f"🤖 AI inference completed")
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'AI inference error: {str(e)}',
                'error': 'AI_INFERENCE_ERROR'
            }), 500

        # Get class names from model
        class_names = model.names if hasattr(model, 'names') else {}
        
        # Count detected objects
        bottle_count = 0
        cap_count = 0
        label_count = 0
        total_detections = 0

        if result.boxes is not None:
            for box in result.boxes:
                total_detections += 1
                class_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                class_name = class_names.get(class_id, str(class_id)).lower()
                
                # นับจำนวนแต่ละประเภท
                if class_name in ["bottle", "ขวด"]:
                    bottle_count += 1
                elif class_name in ["cap", "ฝา"]:
                    cap_count += 1
                elif class_name in ["label", "สลาก"]:
                    label_count += 1

        # คำนวณคะแนน
        score = calculate_score(bottle_count, cap_count, label_count)
        
        logger.info(f"🎯 Detection results - Bottles: {bottle_count}, Caps: {cap_count}, Labels: {label_count}, Score: {score}")

        return jsonify({
            'success': True,
            'message': 'Image analysis completed',
            'result': {
                'bottle_count': bottle_count,
                'cap_count': cap_count,
                'label_count': label_count,
                'total_detections': total_detections,
                'score': score,
                'image_size': f"{image.size[0]}x{image.size[1]}"
            }
        })

    except Exception as e:
        logger.error(f"❌ Unexpected error in scan endpoint: {e}")
        return jsonify({
            'success': False, 
            'message': f'Internal server error: {str(e)}',
            'error': 'INTERNAL_ERROR'
        }), 500

@app.route('/api/ping', methods=['GET'])
def ping():
    """
    ตรวจสอบสถานะของ API และโมเดล
    """
    try:
        model_status = "loaded" if model is not None else "not_loaded"
        model_path_exists = os.path.exists(MODEL_PATH)
        
        return jsonify({
            'success': True,
            'message': 'API is running',
            'status': {
                'api': 'online',
                'model': model_status,
                'model_path': MODEL_PATH,
                'model_file_exists': model_path_exists
            }
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Health check failed: {str(e)}',
            'error': 'HEALTH_CHECK_ERROR'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    ข้อมูลเกี่ยวกับโมเดล AI
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'message': 'Model not loaded'
            }), 500
            
        # ดึงข้อมูลโมเดล
        class_names = model.names if hasattr(model, 'names') else {}
        
        return jsonify({
            'success': True,
            'model_info': {
                'type': 'YOLO',
                'path': MODEL_PATH,
                'classes': class_names,
                'num_classes': len(class_names)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error getting model info: {str(e)}'
        }), 500

def calculate_score(bottle_count, cap_count, label_count):
    """
    คำนวณคะแนนจากจำนวนที่ตรวจพบ
    สามารถปรับสูตรได้ตามต้องการ
    """
    # สูตรตัวอย่าง: ขวดได้คะแนน, ฝาและสลากหักคะแนน
    score = (bottle_count * 50) - (cap_count * 10) - (label_count * 10)
    return max(0, score)  # คะแนนไม่ติดลบ

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False, 
        'message': 'Endpoint not found',
        'error': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False, 
        'message': 'Method not allowed',
        'error': 'METHOD_NOT_ALLOWED'
    }), 405

if __name__ == '__main__':
    print("🚀 Starting Minimal PET Detect API...")
    print(f"📍 Model path: {MODEL_PATH}")
    print(f"🤖 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print("🌐 Available endpoints:")
    print("   - POST /api/scan - Image analysis")
    print("   - GET /api/ping - Health check")
    print("   - GET /api/model-info - Model information")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=True)