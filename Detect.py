import cv2
import torch
import numpy as np
import os

path = 'C:/Users/opal_/OneDrive/Desktop/webcam-test/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

cap = cv2.VideoCapture(0)

threshold = 0.75
dataset_folder = 'C:/Users/opal_/OneDrive/Desktop/webcam-test/Dataset/'
images_folder = os.path.join(dataset_folder, 'images')
labels_folder = os.path.join(dataset_folder, 'labels')

pure_images_folder = 'C:/Users/opal_/OneDrive/Desktop/webcam-test/DetectedImages_Pure'
with_boxes_folder = 'C:/Users/opal_/OneDrive/Desktop/webcam-test/DetectedImages_WithBox'

os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(pure_images_folder, exist_ok=True)
os.makedirs(with_boxes_folder, exist_ok=True)

image_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถจับภาพได้.")
        break

    frame_resized = cv2.resize(frame, (1280, 720))

    result = model(frame_resized)

    pred = result.pred[0]
    pred = pred[pred[:, 4] >= threshold]

    if len(pred) > 0:
        image_counter += 1
        
        image_filename_raw = os.path.join(images_folder, f"image_{image_counter}.jpg")
        cv2.imwrite(image_filename_raw, frame_resized)

        image_filename_pure = os.path.join(pure_images_folder, f"image_{image_counter}.jpg")
        cv2.imwrite(image_filename_pure, frame_resized)

        result.pred[0] = pred
        frame_with_boxes = np.squeeze(result.render())
        image_filename_with_boxes = os.path.join(with_boxes_folder, f"image_with_box_{image_counter}.jpg")
        cv2.imwrite(image_filename_with_boxes, frame_with_boxes)

        label_filename = os.path.join(labels_folder, f"label_{image_counter}.txt")
        with open(label_filename, 'w') as label_file:
            for obj in pred:
                class_id = int(obj[5])
                label_name = model.names[class_id]
                conf = obj[4].item()
                label_file.write(f"{label_name} {conf}\n")

    cv2.imshow("Frame", frame_resized if len(pred) == 0 else frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()