from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# MongoDB Atlas config (ใช้แบบเดียวกับ db.py)
app.config["MONGO_URI"] = "mongodb+srv://s6404062636412:0606@pet.tacvdh9.mongodb.net/pet?retryWrites=true&w=majority&appName=pet"
mongo = PyMongo(app)

# Load YOLO model once at startup
model = YOLO("model-yolov5s/best.pt")  # Adjust path as needed

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing username or password'}), 400
    if mongo.db.users.find_one({'username': username}):  # type: ignore
        return jsonify({'success': False, 'message': 'Username already exists'}), 409
    hashed_pw = generate_password_hash(password)
    mongo.db.users.insert_one({'username': username, 'password': hashed_pw})  # type: ignore
    return jsonify({'success': True, 'message': 'User registered successfully'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing username or password'}), 400
    user = mongo.db.users.find_one({'username': username})  # type: ignore
    if user and check_password_hash(user['password'], password):
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

@app.route('/api/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'}), 400
    image_file = request.files['image']

    # Convert image to numpy array
    image = Image.open(image_file.stream).convert("RGB")
    image_np = np.array(image)

    # Run detection
    results = model(image_np)

    # Count each class
    bottle_count = 0
    cap_count = 0
    label_count = 0

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()
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