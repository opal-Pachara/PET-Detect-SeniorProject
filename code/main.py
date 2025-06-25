import streamlit as st
import os
import sys
from PIL import Image
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the model directory to the path
sys.path.append(str(Path(__file__).parent.parent / "model-yolov5s"))

# Import local modules with aliases to avoid conflicts
from config import AppConfig, ModelConfig, UIConfig, ErrorMessages
from app_utils import ImageUtils, ValidationUtils, ErrorHandler, PerformanceUtils, BottleAnalyzer, ColorAnalyzer

# Custom CSS styles
class Styles:
    """CSS styles for the application."""
    MAIN_HEADER = f"""
    <style>
        .main-header {{
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: {UIConfig.PRIMARY_COLOR};
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: {UIConfig.SECONDARY_COLOR};
            margin-bottom: 1rem;
        }}
        .info-box {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {UIConfig.PRIMARY_COLOR};
            margin: 1rem 0;
        }}
        .upload-area {{
            border: 2px dashed {UIConfig.PRIMARY_COLOR};
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }}
        .metric-card {{
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem;
        }}
    </style>
    """

class SessionState:
    """Manages session state for the application."""
    
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False

class PETDetector:
    """Handles PET object detection using YOLOv5 model."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.model_names = {}
        self.bottle_analyzer = BottleAnalyzer()
    
    def load_model(self) -> bool:
        """Load the YOLOv5 model."""
        try:
            if not ValidationUtils.validate_model_path(self.model_path):
                logger.error(f"Invalid model path: {self.model_path}")
                return False
            
            # Temporarily remove our utils from sys.path to avoid conflicts
            original_path = sys.path.copy()
            if str(Path(__file__).parent) in sys.path:
                sys.path.remove(str(Path(__file__).parent))
            
            try:
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    ModelConfig.MODEL_TYPE, 
                    str(self.model_path), 
                    force_reload=ModelConfig.FORCE_RELOAD
                )
                
                # Test model to ensure it works
                test_input = torch.randn(1, 3, 640, 640)
                with torch.no_grad():
                    test_output = self.model(test_input)
                
            except Exception as model_error:
                logger.error(f"Model loading/initialization error: {model_error}")
                st.error(f"Model loading failed: {str(model_error)}")
                return False
            finally:
                # Restore original path
                sys.path[:] = original_path
            
            if hasattr(self.model, 'names'):
                self.model_names = self.model.names
            else:
                logger.warning("Model does not have 'names' attribute")
                self.model_names = {}
            
            st.session_state.model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            error_msg = ErrorHandler.handle_model_loading_error(e)
            logger.error(f"Error loading model: {e}")
            st.error(error_msg)
            return False
    
    def detect(self, image: np.ndarray, confidence_threshold: float) -> Tuple[np.ndarray, List, Dict]:
        """Perform detection on the given image."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if not ValidationUtils.validate_confidence_threshold(confidence_threshold):
            raise ValueError(f"Invalid confidence threshold: {confidence_threshold}")
        
        try:
            # Convert PIL to OpenCV format if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                img_cv = image
            
            # Perform detection
            results = self.model(img_cv)
            
            # Filter by confidence threshold
            pred = results.pred[0]
            # Convert tensor to numpy if needed
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            pred = pred[pred[:, 4] >= confidence_threshold]
            
            # Create result image with bounding boxes
            if len(pred) > 0:
                # Convert back to tensor for rendering if needed
                if not hasattr(pred, 'cpu'):
                    import torch
                    pred_tensor = torch.from_numpy(pred)
                    results.pred[0] = pred_tensor
                else:
                    results.pred[0] = pred
                result_img = np.squeeze(results.render())
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            else:
                result_img_rgb = image
            
            return result_img_rgb, pred, self.model_names
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise e
    
    def detect_and_analyze_bottles(self, image: np.ndarray, confidence_threshold: float) -> Tuple[np.ndarray, List, Dict]:
        """Perform detection and comprehensive bottle analysis."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Perform basic detection
            result_img, predictions, model_names = self.detect(image, confidence_threshold)
            
            # Analyze each detected bottle
            bottle_analyses = []
            for i, pred in enumerate(predictions):
                try:
                    # Convert tensor to numpy if needed
                    if hasattr(pred, 'cpu'):
                        pred_np = pred.cpu().numpy()
                    else:
                        pred_np = pred
                        
                    class_id = int(pred_np[5])
                    label_name = model_names.get(class_id, f"Class {class_id}")
                    
                    # Only analyze bottles
                    if label_name.lower() in ["bottle", "ขวด"]:
                        # Extract bounding box
                        x1, y1, x2, y2 = pred_np[:4].astype(int)
                        bbox = (x1, y1, x2, y2)
                        
                        # Perform comprehensive bottle analysis
                        analysis = self.bottle_analyzer.analyze_bottle(image, bbox)
                        analysis['prediction_index'] = i
                        analysis['confidence'] = pred_np[4] if isinstance(pred_np[4], (int, float)) else float(pred_np[4])
                        analysis['label_name'] = label_name
                        
                        bottle_analyses.append(analysis)
                        
                except Exception as bottle_error:
                    logger.error(f"Error analyzing bottle {i}: {bottle_error}")
                    continue
            
            return result_img, predictions, model_names, bottle_analyses
            
        except Exception as e:
            logger.error(f"Bottle analysis error: {e}")
            raise e

class ScoreCalculator:
    """Calculates scores based on detected objects."""
    
    @staticmethod
    def calculate_score(predictions: List, model_names: Dict) -> Tuple[int, Dict[str, int]]:
        """Calculate score based on detected objects."""
        bottle_count = 0
        cap_count = 0
        label_count = 0
        
        for obj in predictions:
            # Convert tensor to numpy if needed
            if hasattr(obj, 'cpu'):
                obj_np = obj.cpu().numpy()
            else:
                obj_np = obj
                
            class_id = int(obj_np[5])
            label_name = model_names.get(class_id, f"Class {class_id}")
            label_name_lower = label_name.lower()
            
            if label_name_lower in ModelConfig.CLASS_NAMES['bottle']:
                bottle_count += 1
            elif label_name_lower in ModelConfig.CLASS_NAMES['cap']:
                cap_count += 1
            elif label_name_lower in ModelConfig.CLASS_NAMES['label']:
                label_count += 1
        
        score = (bottle_count * AppConfig.BOTTLE_SCORE_WEIGHT) - \
                (cap_count * AppConfig.CAP_PENALTY_WEIGHT) - \
                (label_count * AppConfig.LABEL_PENALTY_WEIGHT)
        score = max(0, score)
        
        counts = {
            "bottle": bottle_count,
            "cap": cap_count,
            "label": label_count
        }
        
        return score, counts

class UIComponents:
    """UI components for the application."""
    
    @staticmethod
    def render_sidebar() -> Tuple[float, str, float]:
        """Render the sidebar and return confidence threshold and selected page."""
        with st.sidebar:
            st.markdown(f"## {UIConfig.ICONS['home']} {AppConfig.APP_NAME}")
            st.markdown("---")
            
            page = st.radio(
                "📱 Navigation",
                ["🏠 Home", "📸 Upload & Detect", "🔍 Advanced Analysis", "📊 Data Analysis", "ℹ️ About"],
                index=0
            )
            
            st.markdown("---")
            st.markdown("### Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=AppConfig.MIN_CONFIDENCE_THRESHOLD,
                max_value=AppConfig.MAX_CONFIDENCE_THRESHOLD,
                value=AppConfig.DEFAULT_CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Minimum confidence level for detection"
            )
            
            # Calibration settings
            st.markdown("### 📏 Calibration")
            reference_width = st.number_input(
                "Reference Object Width (mm)",
                min_value=10.0,
                max_value=500.0,
                value=100.0,
                step=10.0,
                help="Width of a known object in the image for measurement calibration"
            )
            
            st.markdown("---")
            st.markdown("### System Info")
            st.info(f"Model: {ModelConfig.MODEL_NAME} Custom")
            st.info(f"Threshold: {confidence_threshold:.2f}")
            
            return confidence_threshold, page, reference_width
    
    @staticmethod
    def render_home_page():
        """Render the home page."""
        st.markdown(f'<h1 class="main-header">{UIConfig.ICONS["home"]} {AppConfig.APP_NAME}</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h3>Welcome to {AppConfig.APP_NAME}!</h3>
                <p>{AppConfig.APP_DESCRIPTION}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Features section
        st.markdown('<h2 class="sub-header">✨ Key Features</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(AppConfig.MAX_COLUMNS)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{UIConfig.ICONS['upload']} Image Upload</h3>
                <p>Upload images for PET detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{UIConfig.ICONS['detect']} Real-time Detection</h3>
                <p>Advanced {ModelConfig.MODEL_NAME} model for accurate detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{UIConfig.ICONS['analysis']} Analysis</h3>
                <p>Detailed detection results and statistics</p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_upload_detect_page(confidence_threshold: float):
        """Render the upload and detect page."""
        st.markdown(f'<h1 class="sub-header">{UIConfig.ICONS["upload"]} Upload & Detect PET Objects</h1>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=AppConfig.SUPPORTED_IMAGE_TYPES,
            help="Upload an image to detect PET objects"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Validate and process uploaded image
            try:
                image = Image.open(uploaded_file)
                
                # Validate image
                if not ImageUtils.validate_image(image):
                    st.error(ErrorMessages.IMAGE_ERRORS['invalid_dimensions'])
                    return
                
                # Convert to RGB if needed
                image = ImageUtils.convert_to_rgb(image)
                
                # Resize if too large
                if image.size[0] > AppConfig.MAX_IMAGE_DIMENSIONS[0] or image.size[1] > AppConfig.MAX_IMAGE_DIMENSIONS[1]:
                    image = ImageUtils.resize_image(image, AppConfig.RECOMMENDED_IMAGE_DIMENSIONS)
                    st.warning("Image was resized for better processing performance.")
                
                st.session_state.uploaded_image = image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{UIConfig.ICONS['upload']} Uploaded Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.subheader(f"{UIConfig.ICONS['detect']} Detection Results")
                    
                    if st.button(f"{UIConfig.ICONS['detect']} Start Detection", type="primary"):
                        UIComponents._perform_detection(image, confidence_threshold)
                        
            except Exception as e:
                st.error(ErrorMessages.IMAGE_ERRORS['processing_failed'])
                logger.error(f"Error processing uploaded image: {e}")
    
    @staticmethod
    def _perform_detection(image: Image.Image, confidence_threshold: float):
        """Perform detection on the uploaded image."""
        with st.spinner("Processing image..."):
            try:
                # Load model
                detector = PETDetector(AppConfig.MODEL_PATH)
                if not detector.load_model():
                    return
                
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Perform detection
                result_img, predictions, model_names = detector.detect(img_array, confidence_threshold)
                
                if len(predictions) > 0:
                    st.session_state.detection_results = {
                        'image': result_img,
                        'predictions': predictions,
                        'model_names': model_names
                    }
                    
                    st.success(f"{UIConfig.ICONS['success']} Detected {len(predictions)} PET object(s)!")
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                    
                    # Display detection details
                    UIComponents._display_detection_details(predictions, model_names)
                    
                    # Calculate and display score
                    score, counts = ScoreCalculator.calculate_score(predictions, model_names)
                    st.info(f"คะแนนที่ได้: {score}")
                    st.write(f"ขวด: {counts['bottle']}, ฝา: {counts['cap']}, สลาก: {counts['label']}")
                else:
                    st.warning(f"{UIConfig.ICONS['warning']} {ErrorMessages.DETECTION_ERRORS['no_objects']}")
                    
            except Exception as e:
                error_msg = ErrorHandler.handle_detection_error(e)
                st.error(f"{UIConfig.ICONS['error']} {error_msg}")
                logger.error(f"Detection error: {e}")
    
    @staticmethod
    def _display_detection_details(predictions: List, model_names: Dict):
        """Display detection details."""
        st.subheader("📋 Detection Details")
        for i, obj in enumerate(predictions):
            # Convert tensor to numpy if needed
            if hasattr(obj, 'cpu'):
                obj_np = obj.cpu().numpy()
            else:
                obj_np = obj
                
            class_id = int(obj_np[5])
            label_name = model_names.get(class_id, f"Class {class_id}")
            confidence = obj_np[4] if isinstance(obj_np[4], (int, float)) else obj_np[4].item()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Object", f"#{i+1}")
            with col2:
                st.metric("Class", label_name)
            with col3:
                st.metric("Confidence", PerformanceUtils.format_confidence(confidence))
    
    @staticmethod
    def render_data_analysis_page():
        """Render the data analysis page."""
        st.markdown(f'<h1 class="sub-header">{UIConfig.ICONS["analysis"]} Data Analysis</h1>', unsafe_allow_html=True)
        
        if st.session_state.detection_results is not None:
            st.subheader("📈 Detection Statistics")
            
            predictions = st.session_state.detection_results['predictions']
            model_names = st.session_state.detection_results['model_names']
            
            # Count objects by class
            class_counts = {}
            for obj in predictions:
                # Convert tensor to numpy if needed
                if hasattr(obj, 'cpu'):
                    obj_np = obj.cpu().numpy()
                else:
                    obj_np = obj
                    
                class_id = int(obj_np[5])
                class_name = model_names.get(class_id, f"Class {class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Objects", len(predictions))
            
            with col2:
                # Calculate average confidence
                confidences = []
                for obj in predictions:
                    if hasattr(obj, 'cpu'):
                        obj_np = obj.cpu().numpy()
                    else:
                        obj_np = obj
                    confidence = obj_np[4] if isinstance(obj_np[4], (int, float)) else obj_np[4].item()
                    confidences.append(confidence)
                avg_confidence = np.mean(confidences)
                st.metric("Avg Confidence", PerformanceUtils.format_confidence(avg_confidence))
            
            with col3:
                st.metric("Classes Detected", len(class_counts))
            
            # Class distribution
            st.subheader("📊 Class Distribution")
            for class_name, count in class_counts.items():
                st.progress(count / len(predictions))
                st.write(f"{class_name}: {count} objects")
                
        else:
            st.info(f"{UIConfig.ICONS['info']} No detection data available. Please upload and detect an image first.")
    
    @staticmethod
    def render_about_page():
        """Render the about page."""
        st.markdown(f'<h1 class="sub-header">{UIConfig.ICONS["about"]} About {AppConfig.APP_NAME}</h1>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h3>🎯 Project Overview</h3>
            <p>{AppConfig.APP_DESCRIPTION}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔧 Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Architecture:**
            - {ModelConfig.MODEL_NAME} (You Only Look Once)
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
        
        st.subheader("📋 Version Information")
        st.info(f"Version: {AppConfig.APP_VERSION}")

    @staticmethod
    def render_advanced_analysis_page(confidence_threshold: float, reference_width: float):
        """Render the advanced bottle analysis page."""
        st.markdown(f'<h1 class="sub-header">{UIConfig.ICONS["detect"]} Advanced Bottle Analysis</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>🔍 Advanced Analysis Features</h3>
            <p>This page provides comprehensive bottle analysis including color segmentation, transparency detection, and precise measurements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file for advanced analysis",
            type=AppConfig.SUPPORTED_IMAGE_TYPES,
            help="Upload an image to perform advanced bottle analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Validate and process uploaded image
            try:
                image = Image.open(uploaded_file)
                
                # Validate image
                if not ImageUtils.validate_image(image):
                    st.error(ErrorMessages.IMAGE_ERRORS['invalid_dimensions'])
                    return
                
                # Convert to RGB if needed
                image = ImageUtils.convert_to_rgb(image)
                
                # Resize if too large
                if image.size[0] > AppConfig.MAX_IMAGE_DIMENSIONS[0] or image.size[1] > AppConfig.MAX_IMAGE_DIMENSIONS[1]:
                    image = ImageUtils.resize_image(image, AppConfig.RECOMMENDED_IMAGE_DIMENSIONS)
                    st.warning("Image was resized for better processing performance.")
                
                st.session_state.uploaded_image = image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{UIConfig.ICONS['upload']} Uploaded Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.subheader(f"{UIConfig.ICONS['detect']} Advanced Analysis Results")
                    
                    if st.button(f"{UIConfig.ICONS['detect']} Start Advanced Analysis", type="primary"):
                        UIComponents._perform_advanced_analysis(image, confidence_threshold, reference_width)
                        
            except Exception as e:
                st.error(ErrorMessages.IMAGE_ERRORS['processing_failed'])
                logger.error(f"Error processing uploaded image: {e}")
    
    @staticmethod
    def _perform_advanced_analysis(image: Image.Image, confidence_threshold: float, reference_width: float):
        """Perform advanced bottle analysis."""
        with st.spinner("Performing advanced analysis..."):
            try:
                # Load model
                detector = PETDetector(AppConfig.MODEL_PATH)
                if not detector.load_model():
                    return
                
                # Convert PIL to numpy array
                img_array = np.array(image)
                
                # Perform advanced detection and analysis
                result_img, predictions, model_names, bottle_analyses = detector.detect_and_analyze_bottles(img_array, confidence_threshold)
                
                if len(predictions) > 0:
                    st.session_state.detection_results = {
                        'image': result_img,
                        'predictions': predictions,
                        'model_names': model_names,
                        'bottle_analyses': bottle_analyses
                    }
                    
                    st.success(f"{UIConfig.ICONS['success']} Detected {len(predictions)} object(s)!")
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                    
                    # Display advanced analysis results
                    if bottle_analyses:
                        UIComponents._display_advanced_analysis_results(bottle_analyses, reference_width)
                    else:
                        st.info("No bottles detected for advanced analysis.")
                    
                    # Display basic detection details
                    UIComponents._display_detection_details(predictions, model_names)
                    
                    # Calculate and display score
                    score, counts = ScoreCalculator.calculate_score(predictions, model_names)
                    st.info(f"คะแนนที่ได้: {score}")
                    st.write(f"ขวด: {counts['bottle']}, ฝา: {counts['cap']}, สลาก: {counts['label']}")
                else:
                    st.warning(f"{UIConfig.ICONS['warning']} {ErrorMessages.DETECTION_ERRORS['no_objects']}")
                    
            except Exception as e:
                error_msg = ErrorHandler.handle_detection_error(e)
                st.error(f"{UIConfig.ICONS['error']} {error_msg}")
                logger.error(f"Advanced analysis error: {e}")
    
    @staticmethod
    def _display_advanced_analysis_results(bottle_analyses: List[Dict], reference_width: float):
        """Display advanced bottle analysis results."""
        st.subheader("🔍 Advanced Bottle Analysis")
        
        if not bottle_analyses:
            st.info("No bottles detected for advanced analysis.")
            return
        
        # Summary statistics
        transparent_count = sum(1 for analysis in bottle_analyses if analysis.get('color', {}).get('is_transparent', False))
        colored_count = len(bottle_analyses) - transparent_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bottles", len(bottle_analyses))
        with col2:
            st.metric("Transparent", transparent_count)
        with col3:
            st.metric("Colored", colored_count)
        
        # Detailed analysis for each bottle
        st.subheader("📋 Detailed Bottle Analysis")
        
        for i, analysis in enumerate(bottle_analyses):
            try:
                with st.expander(f"Bottle #{i+1} - {analysis.get('label_name', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎨 Color Analysis**")
                        color_info = analysis.get('color', {})
                        
                        # Color display
                        if 'rgb' in color_info:
                            r, g, b = color_info['rgb']
                            st.markdown(f"**Color:** {color_info.get('color_name', 'Unknown')}")
                            st.markdown(f"**RGB:** ({r}, {g}, {b})")
                            
                            # Display color swatch
                            color_swatch = np.full((50, 100, 3), [r, g, b], dtype=np.uint8)
                            st.image(color_swatch, caption="Color", width=100)
                        
                        st.markdown(f"**Transparent:** {'Yes' if color_info.get('is_transparent', False) else 'No'}")
                        if 'transparent_percentage' in color_info:
                            st.markdown(f"**Transparency:** {color_info['transparent_percentage']:.1f}%")
                    
                    with col2:
                        st.markdown("**📏 Size Analysis**")
                        size_info = analysis.get('size', {})
                        
                        if size_info.get('calibrated', False):
                            st.metric("Width", f"{size_info.get('width_mm', 0):.1f} mm")
                            st.metric("Height", f"{size_info.get('height_mm', 0):.1f} mm")
                            st.metric("Volume", f"{size_info.get('volume_ml', 0):.1f} ml")
                            st.markdown(f"**Size Category:** {size_info.get('size_category', 'Unknown')}")
                        else:
                            st.metric("Width", f"{size_info.get('width_pixels', 0)} px")
                            st.metric("Height", f"{size_info.get('height_pixels', 0)} px")
                            st.info("⚠️ Calibration needed for real-world measurements")
                    
                    # Bottle classification
                    bottle_analyzer = BottleAnalyzer()
                    bottle_type = bottle_analyzer.classify_bottle_type(analysis)
                    st.markdown(f"**🏷️ Classification:** {bottle_type}")
                    
                    # Confidence
                    confidence = analysis.get('confidence', 0)
                    st.progress(confidence)
                    st.markdown(f"**Confidence:** {PerformanceUtils.format_confidence(confidence)}")
                    
            except Exception as bottle_display_error:
                logger.error(f"Error displaying bottle {i}: {bottle_display_error}")
                st.error(f"Error displaying bottle {i+1}")
                continue
        
        # Calibration information
        if reference_width > 0:
            st.info(f"📏 Reference width set to {reference_width} mm. Use this for accurate measurements.")
        else:
            st.warning("⚠️ Set reference width for accurate measurements.")

class PETDetectionApp:
    """Main application class for PET Detection System."""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_styles()
        SessionState.initialize()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title=AppConfig.PAGE_TITLE,
            page_icon=AppConfig.PAGE_ICON,
            layout=AppConfig.LAYOUT,
            initial_sidebar_state=AppConfig.INITIAL_SIDEBAR_STATE
        )
    
    def setup_styles(self):
        """Setup custom CSS styles."""
        st.markdown(Styles.MAIN_HEADER, unsafe_allow_html=True)
    
    def run(self):
        """Run the main application."""
        confidence_threshold, page, reference_width = UIComponents.render_sidebar()
        
        if "🏠 Home" in page:
            UIComponents.render_home_page()
        elif "📸 Upload & Detect" in page:
            UIComponents.render_upload_detect_page(confidence_threshold)
        elif "🔍 Advanced Analysis" in page:
            UIComponents.render_advanced_analysis_page(confidence_threshold, reference_width)
        elif "📊 Data Analysis" in page:
            UIComponents.render_data_analysis_page()
        elif "ℹ️ About" in page:
            UIComponents.render_about_page()

def main():
    """Main function to run the application."""
    try:
        app = PETDetectionApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"{UIConfig.ICONS['error']} An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
