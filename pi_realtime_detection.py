#!/usr/bin/env python3
"""
Real-time PET Detection for Raspberry Pi
เปิดกล้องแบบ Real-time และแสดง Class ทันทีที่เจอขวด
"""

import cv2
import torch
import numpy as np
import os
import time

# ใช้ path ที่เหมาะสมกับ Raspberry Pi
model_path = '/home/pi/PET-Detect-SeniorProject/model-yolov11/best.pt'

# โหลดโมเดล
print("กำลังโหลดโมเดล...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=True)
    print("โหลดโมเดลสำเร็จ")
except Exception as e:
    print(f"ไม่สามารถโหลดโมเดลได้: {e}")
    exit(1)

# ตั้งค่า threshold
threshold = 0.6

# สร้างโฟลเดอร์สำหรับบันทึกภาพ
output_folder = '/home/pi/PET-Detect-SeniorProject/detected_images'
os.makedirs(output_folder, exist_ok=True)

# เปิดกล้อง
print("กำลังเปิดกล้อง...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit(1)

# ตั้งค่ากล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("เปิดกล้องสำเร็จ")
print("เริ่มการตรวจจับ Real-time...")
print("กด 'q' เพื่อออก")

image_counter = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ไม่สามารถจับภาพได้")
            break
        
        # ปรับขนาดภาพ
        frame_resized = cv2.resize(frame, (640, 480))
        
        # ส่งไป AI Model
        result = model(frame_resized)
        
        # กรองผลลัพธ์ตาม confidence
        pred = result.pred[0]
        pred = pred[pred[:, 4] >= threshold]
        
        # ถ้าเจอวัตถุ
        if len(pred) > 0:
            image_counter += 1
            
            # บันทึกภาพต้นฉบับ
            timestamp = int(time.time())
            image_filename = os.path.join(output_folder, f"detected_{timestamp}.jpg")
            cv2.imwrite(image_filename, frame_resized)
            
            # แสดงผลบนหน้าจอ
            frame_with_boxes = np.squeeze(result.render())
            
            # แสดงข้อมูล Class ที่เจอ
            print(f"\nตรวจจับได้ {len(pred)} วัตถุ:")
            for i, obj in enumerate(pred):
                class_id = int(obj[5])
                label_name = model.names[class_id]
                conf = obj[4].item()
                print(f"   {i+1}. {label_name} (ความมั่นใจ: {conf:.2f})")
            
            print(f"บันทึกภาพ: {image_filename}")
            
            # แสดงภาพพร้อม Bounding Box
            cv2.imshow("PET Detection - Real-time", frame_with_boxes)
        else:
            # แสดงภาพปกติ
            cv2.imshow("PET Detection - Real-time", frame_resized)
        
        # ตรวจสอบการกดปุ่ม
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("หยุดการทำงาน")
            break
        elif key == ord('s'):
            # บันทึกภาพปัจจุบัน
            timestamp = int(time.time())
            image_filename = os.path.join(output_folder, f"manual_{timestamp}.jpg")
            cv2.imwrite(image_filename, frame_resized)
            print(f"บันทึกภาพด้วยตนเอง: {image_filename}")

except KeyboardInterrupt:
    print("\nหยุดการทำงานด้วย Ctrl+C")

finally:
    # ปิดกล้องและหน้าต่าง
    cap.release()
    cv2.destroyAllWindows()
    print("ปิดกล้องและหน้าต่างแล้ว")

print(f"สรุป: บันทึกภาพทั้งหมด {image_counter} ภาพ")
print(f"ภาพทั้งหมดอยู่ใน: {output_folder}")
