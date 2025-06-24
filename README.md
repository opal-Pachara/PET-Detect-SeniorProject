# 🐾 PET Detection System

ระบบตรวจจับ PET (Polyethylene Terephthalate) ด้วยเทคโนโลยี Computer Vision และ Deep Learning

## ✨ Features

- 📸 **อัปโหลดรูปภาพ** - รองรับไฟล์ PNG, JPG, JPEG
- 🔍 **ตรวจจับแบบ Real-time** - ใช้โมเดล YOLOv5s ที่เทรนมาสำหรับ PET
- 📊 **วิเคราะห์ข้อมูล** - แสดงสถิติและรายละเอียดการตรวจจับ
- 🎛️ **ปรับแต่งความแม่นยำ** - ปรับ Confidence Threshold ได้
- 🎨 **UI สวยงาม** - หน้าตาใช้งานง่ายและทันสมัย

## 🚀 การติดตั้ง

### 1. Clone Repository
```bash
git clone <repository-url>
cd PET-Detect-SeniorProject
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. ตรวจสอบไฟล์โมเดล
ให้แน่ใจว่ามีไฟล์ `best.pt` อยู่ในโฟลเดอร์ `model-yolov5s/`

## 🎯 การใช้งาน

### รันแอปพลิเคชัน
```bash
cd code
streamlit run main.py
```

### หน้าจอหลัก
1. **🏠 Home** - หน้าหลักแสดงข้อมูลและฟีเจอร์
2. **📸 Upload & Detect** - อัปโหลดรูปภาพและตรวจจับ PET
3. **📊 Data Analysis** - วิเคราะห์ผลการตรวจจับ
4. **ℹ️ About** - ข้อมูลเกี่ยวกับโปรเจค

### ขั้นตอนการตรวจจับ
1. ไปที่หน้า "📸 Upload & Detect"
2. อัปโหลดรูปภาพที่ต้องการตรวจจับ
3. ปรับ Confidence Threshold ตามต้องการ
4. กดปุ่ม "🚀 Start Detection"
5. ดูผลลัพธ์การตรวจจับ

## 📁 โครงสร้างโปรเจค

```
PET-Detect-SeniorProject/
├── code/
│   ├── main.py              # ไฟล์หลักของ Streamlit app
│   └── data_model.py        # ไฟล์สำหรับจัดการข้อมูล
├── model-yolov5s/
│   ├── best.pt              # โมเดล YOLOv5s ที่เทรนแล้ว
│   └── Detect.py            # สคริปต์สำหรับตรวจจับด้วย webcam
├── requirements.txt         # Dependencies
└── README.md               # คู่มือการใช้งาน
```

## 🔧 Technical Details

### เทคโนโลยีที่ใช้
- **Python 3.x** - ภาษาหลัก
- **Streamlit** - Web framework
- **PyTorch** - Deep learning framework
- **YOLOv5** - Object detection model
- **OpenCV** - Computer vision library
- **Pillow** - Image processing

### โมเดล
- **Architecture**: YOLOv5s
- **Custom Training**: เทรนด้วยชุดข้อมูล PET
- **Performance**: Real-time detection

## 📊 ผลลัพธ์

ระบบจะแสดง:
- รูปภาพต้นฉบับ
- รูปภาพที่มี bounding boxes
- จำนวนวัตถุที่ตรวจจับได้
- ความแม่นยำ (Confidence) ของแต่ละการตรวจจับ
- สถิติการวิเคราะห์

## 🛠️ การปรับแต่ง

### ปรับ Confidence Threshold
- ใช้ slider ใน sidebar
- ค่าตั้งแต่ 0.1 ถึง 1.0
- ค่ายิ่งสูงยิ่งแม่นยำแต่จะตรวจจับได้น้อยลง

### เพิ่มคลาสใหม่
1. เทรนโมเดลใหม่ด้วยข้อมูลคลาสที่ต้องการ
2. แทนที่ไฟล์ `best.pt`
3. อัปเดตโค้ดตามความต้องการ

## 🐛 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **โมเดลไม่โหลด**
   - ตรวจสอบว่ามีไฟล์ `best.pt` ในโฟลเดอร์ `model-yolov5s/`
   - ตรวจสอบ path ในโค้ด

2. **Dependencies ไม่ครบ**
   - รัน `pip install -r requirements.txt`
   - ตรวจสอบเวอร์ชัน Python (แนะนำ 3.8+)

3. **การตรวจจับไม่แม่นยำ**
   - ลด Confidence Threshold
   - ตรวจสอบคุณภาพรูปภาพ
   - เทรนโมเดลใหม่ด้วยข้อมูลเพิ่มเติม

## 📞 การติดต่อ

หากมีปัญหาหรือข้อสงสัย กรุณาติดต่อทีมพัฒนา

---

**หมายเหตุ**: โปรเจคนี้พัฒนาขึ้นเพื่อการศึกษาและวิจัยในด้าน Computer Vision และ Machine Learning 