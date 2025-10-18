<<<<<<< HEAD
=======
import streamlit as st
>>>>>>> fd413dfd8328b8377909f567fd5b74cd5d59ac37
import cv2
import torch
import numpy as np
import os
<<<<<<< HEAD

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
=======
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="PET Bottle Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if 'detection_counts' not in st.session_state:
    st.session_state.detection_counts = defaultdict(int)
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best.pt')
    return torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=True)

def process_frame(frame, model, confidence_threshold=0.75):
    frame_resized = cv2.resize(frame, (1280, 720))
    results = model(frame_resized)
    
    # Filter predictions by confidence threshold
    pred = results.pred[0]
    pred = pred[pred[:, 4] >= confidence_threshold]
    
    # Draw boxes and get detections
    detections = []
    if len(pred) > 0:
        for *box, conf, cls in pred:
            label = model.names[int(cls)]
            confidence = float(conf)
            detections.append({
                'label': label,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Update session state counts
            st.session_state.detection_counts[label] += 1
            st.session_state.detection_history.append({
                'label': label,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
    
    return np.squeeze(results.render()), detections

def create_detection_chart():
    df = pd.DataFrame(list(st.session_state.detection_counts.items()), 
                     columns=['Object', 'Count'])
    fig = px.bar(df, x='Object', y='Count',
                 title='Total Detections by Object Type',
                 color='Object')
    return fig

def create_confidence_histogram(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = px.histogram(df, x='confidence', 
                      color='label',
                      title='Detection Confidence Distribution',
                      nbins=20)
    return fig

# Main app layout
st.title("🔍 PET Bottle Detection Dashboard")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75)
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], index=0)

# Initialize model
@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    model_names = model.names if hasattr(model, 'names') else ['PET Bottle']
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None
    model_names = ['PET Bottle']

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Live detection view
    st.subheader("Live Detection")
    frame_placeholder = st.empty()
    
    # Detection statistics
    st.subheader("Detection Statistics")
    stats_cols = st.columns(len(model_names))
    for idx, name in enumerate(model_names):
        with stats_cols[idx]:
            st.metric(str(name), st.session_state.detection_counts[name])

with col2:
    # Detection charts
    st.plotly_chart(create_detection_chart(), use_container_width=True)
    
    if st.session_state.detection_history:
        conf_hist = create_confidence_histogram(st.session_state.detection_history)
        if conf_hist:
            st.plotly_chart(conf_hist, use_container_width=True)

# Camera feed and detection loop
cap = cv2.VideoCapture(camera_index)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error accessing the camera feed.")
            break
            
        # Process frame
        processed_frame, detections = process_frame(frame, model, confidence_threshold)
        
        # Convert BGR to RGB
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update frame in the app
        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
        
        # Break if stop button is pressed
        if st.sidebar.button("Stop"):
            break
            
finally:
    cap.release()
>>>>>>> fd413dfd8328b8377909f567fd5b74cd5d59ac37
