from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# MongoDB Atlas config (ใช้แบบเดียวกับ db.py)
app.config["MONGO_URI"] = "mongodb+srv://s6404062636412:0606@pet.tacvdh9.mongodb.net/pet?retryWrites=true&w=majority&appName=pet"
mongo = PyMongo(app)

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
    # รับไฟล์ภาพจาก frontend (multipart/form-data)
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'}), 400
    image = request.files['image']
    # TODO: นำภาพไปตรวจจับด้วยโมเดล (mockup ตอบกลับผลลัพธ์ตัวอย่าง)
    # สามารถต่อยอดให้เรียก YOLOv5 ได้ภายหลัง
    return jsonify({
        'success': True,
        'message': 'Scan completed (mockup)',
        'result': {
            'bottle_detected': True,
            'score': 85,
            'details': 'This is a mockup result.'
        }
    })

@app.route('/api/ping', methods=['GET'])
def ping():
    try:
        # ลองนับจำนวน user ใน collection (หรือจะใช้ mongo.cx.server_info() ก็ได้)
        mongo.db.users.count_documents({})  # type: ignore
        return jsonify({'success': True, 'message': 'MongoDB connected'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'MongoDB connection error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 