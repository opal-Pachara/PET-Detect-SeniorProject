from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import os
# import torch  # ไม่ใช้ torch.hub แล้ว
from PIL import Image
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO  # เพิ่มบรรทัดนี้

app = Flask(__name__)
CORS(app)

# PostgreSQL configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'pet_detect_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'password')
}

# Load YOLOv11n custom model once at startup (use ultralytics)
model = YOLO('model-yolov5s/best.pt')  # Custom YOLOv11n trained model

def get_db_connection():
    """สร้างการเชื่อมต่อฐานข้อมูล PostgreSQL"""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing username or password'}), 400
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM members WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409
        
        hashed_pw = generate_password_hash(password)
        cursor.execute("INSERT INTO members (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
        connection.commit()
        cursor.close()
        connection.close()
        
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing username or password'}), 400
    
    connection = get_db_connection()
    if not connection:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM members WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if user and check_password_hash(user['password_hash'], password):
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    except psycopg2.Error as e:
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500

@app.route('/api/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'}), 400
    image_file = request.files['image']

    # Convert image to numpy array
    image = Image.open(image_file.stream).convert("RGB")
    image_np = np.array(image)

    # Run detection (ultralytics YOLO)
    results = model(image_np)
    result = results[0]  # ultralytics v8+ ผลลัพธ์เป็น list

    # Get class names from model
    class_names = model.names if hasattr(model, 'names') else {}

    # Count each class
    bottle_count = 0
    cap_count = 0
    label_count = 0

    # วนลูปผ่าน result.boxes (ultralytics YOLOv8+)
    for box in result.boxes:
        class_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
        class_name = class_names.get(class_id, str(class_id)).lower()
        if class_name in ["bottle", "ขวด"]:
            bottle_count += 1
        elif class_name in ["cap", "ฝา"]:
            cap_count += 1
        elif class_name in ["label", "สลาก"]:
            label_count += 1

    # Example: Calculate score (ปรับสูตรได้)
    score = (bottle_count * 50) - (cap_count * 10) - (label_count * 10)
    score = max(0, score)

    return jsonify({
        'success': True,
        'message': 'Scan completed',
        'result': {
            'bottle_count': bottle_count,
            'cap_count': cap_count,
            'label_count': label_count,
            'score': score
        }
    })

@app.route('/api/ping', methods=['GET'])
def ping():
    try:
        mongo.db.users.count_documents({})  # type: ignore
        return jsonify({'success': True, 'message': 'MongoDB connected'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'MongoDB connection error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 