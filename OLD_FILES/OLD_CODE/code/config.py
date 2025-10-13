"""
Configuration settings for the PET Detection System.
"""

from pathlib import Path
from typing import List, Tuple

class AppConfig:
    """Application configuration settings."""
    
    # Application metadata
    APP_NAME = "PET Detection System"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Advanced PET object detection using YOLOv5"
    
    # Page configuration
    PAGE_TITLE = "PET Detection System"
    PAGE_ICON = "🐾"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "model-yolov5s"
    MODEL_PATH = MODEL_DIR / "best.pt"
    
    # Supported file types
    SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
    
    # Model settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.75
    MIN_CONFIDENCE_THRESHOLD = 0.1
    MAX_CONFIDENCE_THRESHOLD = 1.0
    
    # Image processing settings
    MAX_IMAGE_SIZE_MB = 10.0
    MAX_IMAGE_DIMENSIONS = (4000, 4000)
    RECOMMENDED_IMAGE_DIMENSIONS = (800, 600)
    
    # Score calculation weights
    BOTTLE_SCORE_WEIGHT = 50
    CAP_PENALTY_WEIGHT = 10
    LABEL_PENALTY_WEIGHT = 10
    
    # UI settings
    SIDEBAR_WIDTH = 300
    MAX_COLUMNS = 3
    
    # Navigation options
    NAVIGATION_OPTIONS = [
        "🏠 Home",
        "📸 Upload & Detect", 
        "🔍 Advanced Analysis",
        "📊 Data Analysis",
        "ℹ️ About"
    ]
    
    # Color analysis settings
    COLOR_ANALYSIS = {
        'DEFAULT_N_COLORS': 3,
        'TRANSPARENT_THRESHOLD': 40.0,  # Percentage
        'COLOR_VARIANCE_THRESHOLD': 1000,
        'HSV_TRANSPARENT_LOWER': [0, 0, 200],
        'HSV_TRANSPARENT_UPPER': [180, 30, 255]
    }
    
    # Measurement settings
    MEASUREMENT = {
        'DEFAULT_REFERENCE_WIDTH_MM': 100.0,
        'MIN_REFERENCE_WIDTH_MM': 10.0,
        'MAX_REFERENCE_WIDTH_MM': 500.0,
        'VOLUME_CALCULATION_ENABLED': True
    }
    
    # Bottle size categories (ml)
    BOTTLE_SIZE_CATEGORIES = {
        'VERY_SMALL': 100,
        'SMALL': 300,
        'MEDIUM': 600,
        'LARGE': 1000
    }

class ModelConfig:
    """Model-specific configuration."""
    
    # YOLOv5 settings
    MODEL_NAME = "YOLOv5s"
    MODEL_TYPE = "custom"
    FORCE_RELOAD = True
    
    # Detection settings
    DEFAULT_IOU_THRESHOLD = 0.45
    DEFAULT_AGNOSTIC_NMS = False
    DEFAULT_MAX_DET = 1000
    
    # Class names (Thai and English)
    CLASS_NAMES = {
        'bottle': ['bottle', 'ขวด'],
        'cap': ['cap', 'ฝา'],
        'label': ['label', 'สลาก']
    }

class UIConfig:
    """UI-specific configuration."""
    
    # Color scheme
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#2c3e50"
    SUCCESS_COLOR = "#28a745"
    WARNING_COLOR = "#ffc107"
    ERROR_COLOR = "#dc3545"
    INFO_COLOR = "#17a2b8"
    
    # CSS classes
    CSS_CLASSES = {
        'main_header': 'main-header',
        'sub_header': 'sub_header',
        'info_box': 'info-box',
        'upload_area': 'upload-area',
        'metric_card': 'metric-card'
    }
    
    # Icons
    ICONS = {
        'home': '🏠',
        'upload': '📸',
        'detect': '🔍',
        'analysis': '📊',
        'about': 'ℹ️',
        'settings': '⚙️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': '📝'
    }

class LoggingConfig:
    """Logging configuration."""
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "pet_detection.log"
    
    # Log levels for different components
    COMPONENT_LOG_LEVELS = {
        'main': 'INFO',
        'detector': 'DEBUG',
        'ui': 'INFO',
        'utils': 'DEBUG'
    }

class ErrorMessages:
    """Error message templates."""
    
    MODEL_ERRORS = {
        'not_found': 'Model file not found. Please check if the model file exists in the correct location.',
        'loading_failed': 'Failed to load model. Please check your model file and path.',
        'invalid_model': 'Invalid model file. Please check the model format.',
        'permission_denied': 'Permission denied accessing model file. Please check file permissions.',
        'out_of_memory': 'Out of memory error. Please try with a smaller image or restart the application.'
    }
    
    IMAGE_ERRORS = {
        'invalid_format': 'Invalid image format. Please upload a valid image file.',
        'file_too_large': 'File size too large. Please upload a smaller image.',
        'invalid_dimensions': 'Invalid image dimensions. Please try with a different image.',
        'processing_failed': 'Error processing image. Please try again.',
        'upload_failed': 'Failed to upload image. Please try again.'
    }
    
    DETECTION_ERRORS = {
        'no_objects': 'No objects detected with current confidence threshold.',
        'processing_error': 'Error during detection. Please try again.',
        'gpu_error': 'GPU processing error. Please try with CPU processing or restart the application.',
        'input_error': 'Input error. Please check the image format and try again.'
    }
    
    GENERAL_ERRORS = {
        'unknown': 'An unknown error occurred. Please try again.',
        'network': 'Network error. Please check your internet connection.',
        'timeout': 'Request timeout. Please try again.',
        'server_error': 'Server error. Please try again later.'
    }

class PerformanceConfig:
    """Performance-related configuration."""
    
    # Memory limits
    MAX_MEMORY_USAGE_MB = 2048
    MAX_IMAGE_PIXELS = 16000000  # 4000x4000
    
    # Processing timeouts
    MODEL_LOADING_TIMEOUT = 30  # seconds
    DETECTION_TIMEOUT = 60  # seconds
    
    # Cache settings
    ENABLE_CACHING = True
    CACHE_TTL = 3600  # 1 hour
    
    # Batch processing
    BATCH_SIZE = 1
    ENABLE_BATCH_PROCESSING = False 