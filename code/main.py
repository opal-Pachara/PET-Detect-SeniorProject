import streamlit as st

# ต้องเรียก set_page_config ทันที
st.set_page_config(
    page_title="PET Detection System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ที่เหลือค่อยตามมา
import os
import sys
from PIL import Image
import torch
import cv2
import numpy as np
from pathlib import Path


# Add the model directory to the path
sys.path.append(str(Path(__file__).parent.parent / "model-yolov5s"))


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# ------------------------------
# Sidebar menu
# ------------------------------
with st.sidebar:
    st.markdown("## 🐾 PET Detection System")
    st.markdown("---")
    
    page = st.radio(
        "📱 Navigation",
        ["🏠 Home", "📸 Upload & Detect", "📊 Data Analysis", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Minimum confidence level for detection"
    )
    
    st.markdown("---")
    st.markdown("### System Info")
    st.info(f"Model: YOLOv5s Custom")
    st.info(f"Threshold: {confidence_threshold:.2f}")

# ------------------------------
# Home Page
# ------------------------------
def home_page():
    st.markdown('<h1 class="main-header">🐾 PET Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to PET Detection System!</h3>
            <p>This system uses advanced computer vision and deep learning to detect and classify PET (Polyethylene Terephthalate) objects in images.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown('<h2 class="sub-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📸 Image Upload</h3>
            <p>Upload images for PET detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Real-time Detection</h3>
            <p>Advanced YOLOv5 model for accurate detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analysis</h3>
            <p>Detailed detection results and statistics</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# Upload & Detect Page
# ------------------------------
def upload_detect_page():
    st.markdown('<h1 class="sub-header">📸 Upload & Detect PET Objects</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to detect PET objects"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            if st.button("🚀 Start Detection", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Load model
                        model_path = Path(__file__).parent.parent / "model-yolov5s" / "best.pt"
                        model = torch.hub.load('ultralytics/yolov5', 'custom', str(model_path), force_reload=True)
                        
                        if not callable(model):
                            st.error("Model did not load correctly or is not callable. Please check your model file and path.")
                            st.stop()
                        
                        st.write(f"Loaded model type: {type(model)}")
                        
                        # Convert PIL to OpenCV format
                        img_array = np.array(image)
                        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Perform detection
                        results = model(img_cv)
                        
                        # Filter by confidence threshold
                        pred = results.pred[0]
                        pred = pred[pred[:, 4] >= confidence_threshold]
                        
                        if len(pred) > 0:
                            # Create result image with bounding boxes
                            results.pred[0] = pred
                            result_img = np.squeeze(results.render())
                            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            
                            if hasattr(model, 'names'):
                                model_names = model.names
                            else:
                                st.error("Model does not have attribute 'names'. Please check your model file.")
                                model_names = {}
                            
                            st.session_state.detection_results = {
                                'image': result_img_rgb,
                                'predictions': pred,
                                'model_names': model_names
                            }
                            
                            st.success(f"✅ Detected {len(pred)} PET object(s)!")
                            st.image(result_img_rgb, caption="Detection Results", use_column_width=True)
                            
                            # Display detection details
                            st.subheader("📋 Detection Details")
                            for i, obj in enumerate(pred):
                                class_id = int(obj[5])
                                label_name = model_names.get(class_id, f"Class {class_id}")
                                confidence = obj[4].item()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Object", f"#{i+1}")
                                with col2:
                                    st.metric("Class", label_name)
                                with col3:
                                    st.metric("Confidence", f"{confidence:.3f}")
                            
                            # --- คำนวณคะแนนแบบใหม่ (case-insensitive) ---
                            bottle_count = 0
                            cap_count = 0
                            label_count = 0
                            for obj in pred:
                                class_id = int(obj[5])
                                label_name = model_names.get(class_id, f"Class {class_id}")
                                label_name_lower = label_name.lower()
                                if label_name_lower == "bottle" or label_name_lower == "ขวด":
                                    bottle_count += 1
                                if label_name_lower == "cap" or label_name_lower == "ฝา":
                                    cap_count += 1
                                if label_name_lower == "label" or label_name_lower == "สลาก":
                                    label_count += 1
                            score = (bottle_count * 50) - (cap_count * 10) - (label_count * 10)
                            if score < 0:
                                score = 0
                            st.info(f"คะแนนที่ได้: {score}")
                            st.write(f"ขวด: {bottle_count}, ฝา: {cap_count}, สลาก: {label_count}")
                        else:
                            st.warning("⚠️ No PET objects detected with the current confidence threshold.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during detection: {str(e)}")
                        st.info("Please make sure the model file 'best.pt' is available in the model-yolov5s directory.")

# ------------------------------
# Data Analysis Page
# ------------------------------
def data_analysis_page():
    st.markdown('<h1 class="sub-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.detection_results is not None:
        st.subheader("📈 Detection Statistics")
        
        predictions = st.session_state.detection_results['predictions']
        model_names = st.session_state.detection_results['model_names']
        
        # Count objects by class
        class_counts = {}
        for obj in predictions:
            class_id = int(obj[5])
            class_name = model_names.get(class_id, f"Class {class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Objects", len(predictions))
        
        with col2:
            avg_confidence = np.mean([obj[4].item() for obj in predictions])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            st.metric("Classes Detected", len(class_counts))
        
        # Class distribution
        st.subheader("📊 Class Distribution")
        for class_name, count in class_counts.items():
            st.progress(count / len(predictions))
            st.write(f"{class_name}: {count} objects")
            
    else:
        st.info("📝 No detection data available. Please upload and detect an image first.")

# ------------------------------
# About Page
# ------------------------------
def about_page():
    st.markdown('<h1 class="sub-header">ℹ️ About PET Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This PET Detection System is designed to identify and classify Polyethylene Terephthalate (PET) objects in images using advanced computer vision techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Architecture:**
        - YOLOv5s (You Only Look Once)
        - Custom trained on PET dataset
        - Real-time object detection
        
        **Features:**
        - Image upload and processing
        - Confidence threshold adjustment
        - Bounding box visualization
        - Detection statistics
        """)
    
    with col2:
        st.markdown("""
        **Technologies Used:**
        - Python 3.x
        - PyTorch
        - OpenCV
        - Streamlit
        - YOLOv5
        
        **Performance:**
        - Real-time processing
        - High accuracy detection
        - User-friendly interface
        """)
    
    st.subheader("👥 Development Team")
    st.info("This project was developed as a senior project focusing on computer vision and machine learning applications.")

# ------------------------------
# Page rendering
# ------------------------------
def main():
    if "🏠 Home" in page:
        home_page()
    elif "📸 Upload & Detect" in page:
        upload_detect_page()
    elif "📊 Data Analysis" in page:
        data_analysis_page()
    elif "ℹ️ About" in page:
        about_page()

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":

    main()
