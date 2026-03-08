#!/usr/bin/env python3

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, 'model-yolov11', 'best.pt')

try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Cannot load model: {e}")
    exit(1)

output_folder = 'detected_images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def draw_boxes(frame, results):
    bottles = 0
    cans = 0
    caps = 0
    labels = 0
    
    if len(results) > 0:
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.6:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if class_name in ["Bottle", "bottle"]:
                            bottles += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                            cv2.putText(frame, f"Bottle ({confidence:.2f})", (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        elif class_name in ["Can", "can"]:
                            cans += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                            cv2.putText(frame, f"Can ({confidence:.2f})", (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        elif class_name in ["Cap", "cap"]:
                            caps += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f"Cap ({confidence:.2f})", (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        elif class_name in ["Label", "label"]:
                            labels += 1
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                            cv2.putText(frame, f"Label ({confidence:.2f})", (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 70), (255, 255, 255), 2)
    cv2.putText(frame, f"B: {bottles} | C: {cans}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Cap: {caps} | L: {labels}", (20, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return bottles, cans, caps, labels

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (640, 480))
        frame_display = frame_resized.copy()
        
        results = model(frame_resized, verbose=False)
        bottles, cans, caps, labels = draw_boxes(frame_display, results)
        
        if bottles > 0 or cans > 0 or caps > 0 or labels > 0:
            timestamp = int(time.time())
            image_filename = os.path.join(output_folder, f"detected_{timestamp}.jpg")
            cv2.imwrite(image_filename, frame_resized)
        
        cv2.imshow("PET Detection - Real-time", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            image_filename = os.path.join(output_folder, f"manual_{timestamp}.jpg")
            cv2.imwrite(image_filename, frame_resized)

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()

