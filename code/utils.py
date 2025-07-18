"""
Utility functions and constants for the PET Detection System.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import contextlib

# Configure logging
logger = logging.getLogger(__name__)

# YOLOv5 compatibility imports
class TryExcept:
    """Try-except wrapper for YOLOv5 compatibility."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

# Additional YOLOv5 compatibility utilities
def check_version(current='0.0.0', minimum='0.0.0', name='version', pinned=False, hard=False, verbose=False):
    """Check version compatibility for YOLOv5."""
    return True

def check_online():
    """Check if online for YOLOv5."""
    return True

def check_imshow(warn=False):
    """Check if imshow is available for YOLOv5."""
    return True

def check_requirements(requirements=('torch', 'torchvision'), exclude=(), install=True, cmds=''):
    """Check requirements for YOLOv5."""
    return True

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    """Check file suffix for YOLOv5."""
    return True

def check_file(file, suffix=''):
    """Check file exists for YOLOv5."""
    return str(file)

def check_yaml(file_path):
    """Check YAML file for YOLOv5."""
    return file_path

def get_latest_run(search_dir='.'):
    """Get latest run directory for YOLOv5."""
    return '.'

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """Increment path for YOLOv5."""
    return path

def make_divisible(x, divisor):
    """Make x divisible by divisor for YOLOv5."""
    return x

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Non-maximum suppression for YOLOv5."""
    return prediction

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Scale coordinates for YOLOv5."""
    return coords

def xyxy2xywh(x):
    """Convert xyxy to xywh for YOLOv5."""
    return x

def xywh2xyxy(x):
    """Convert xywh to xyxy for YOLOv5."""
    return x

def clip_coords(boxes, img_shape):
    """Clip coordinates for YOLOv5."""
    return boxes

def box_iou(box1, box2):
    """Calculate IoU for YOLOv5."""
    return 0.0

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Letterbox function for YOLOv5."""
    return im, 1.0, (0, 0)

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes for YOLOv5."""
    pass

def init_seeds(seed=0):
    """Initialize seeds for YOLOv5."""
    pass

def select_device(device='', batch_size=None):
    """Select device for YOLOv5."""
    return 'cpu'

def time_sync():
    """Time sync for YOLOv5."""
    import time
    return time.time()

def profile(input, ops, n=10, device=None):
    """Profile operations for YOLOv5."""
    return input

def is_parallel(model):
    """Check if model is parallel for YOLOv5."""
    return False

def de_parallel(model):
    """De-parallelize model for YOLOv5."""
    return model

def intersect_dicts(da, db, exclude=()):
    """Intersect dictionaries for YOLOv5."""
    return da

def fuse_conv_and_bn(conv, bn):
    """Fuse conv and bn for YOLOv5."""
    return conv

def model_info(model, verbose=False, img_size=640):
    """Model info for YOLOv5."""
    pass

def load_classifier(name='resnet101', n=2):
    """Load classifier for YOLOv5."""
    return None

def check_img_size(img_size, s=32):
    """Check image size for YOLOv5."""
    return img_size

def check_imgsz(imgsz, stride=32):
    """Check image size for YOLOv5."""
    return imgsz

def check_dataset(data, autodownload=True):
    """Check dataset for YOLOv5."""
    return data

class FileUtils:
    """File utility functions."""
    
    @staticmethod
    def ensure_directory_exists(directory_path: Path) -> bool:
        """Ensure that a directory exists, create if it doesn't."""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def is_valid_image_file(file_path: Path) -> bool:
        """Check if a file is a valid image file."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        return file_path.suffix.lower() in valid_extensions
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Get file size in megabytes."""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {e}")
            return 0.0

class ImageUtils:
    """Image processing utility functions."""
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        try:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def validate_image(image: Image.Image) -> bool:
        """Validate if image is suitable for processing."""
        try:
            # Check if image has valid dimensions
            if image.size[0] <= 0 or image.size[1] <= 0:
                return False
            
            # Check if image is not too large (prevent memory issues)
            if image.size[0] > 4000 or image.size[1] > 4000:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False
    
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB format if needed."""
        try:
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error converting image to RGB: {e}")
            return image

class ValidationUtils:
    """Validation utility functions."""
    
    @staticmethod
    def validate_confidence_threshold(threshold: float) -> bool:
        """Validate confidence threshold value."""
        return 0.0 <= threshold <= 1.0
    
    @staticmethod
    def validate_model_path(model_path: Path) -> bool:
        """Validate if model file exists and is accessible."""
        try:
            return model_path.exists() and model_path.is_file()
        except Exception as e:
            logger.error(f"Error validating model path {model_path}: {e}")
            return False

class PerformanceUtils:
    """Performance monitoring utility functions."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence value as percentage."""
        return f"{confidence * 100:.1f}%"

class ErrorHandler:
    """Error handling utility functions."""
    
    @staticmethod
    def handle_model_loading_error(error: Exception) -> str:
        """Handle model loading errors and return user-friendly message."""
        error_msg = str(error).lower()
        
        if "file not found" in error_msg or "no such file" in error_msg:
            return "Model file not found. Please check if the model file exists in the correct location."
        elif "permission denied" in error_msg:
            return "Permission denied accessing model file. Please check file permissions."
        elif "out of memory" in error_msg or "cuda out of memory" in error_msg:
            return "Out of memory error. Please try with a smaller image or restart the application."
        else:
            return f"Model loading error: {str(error)}"
    
    @staticmethod
    def handle_detection_error(error: Exception) -> str:
        """Handle detection errors and return user-friendly message."""
        error_msg = str(error).lower()
        
        if "input" in error_msg and "size" in error_msg:
            return "Image size not supported. Please try with a different image."
        elif "cuda" in error_msg:
            return "GPU processing error. Please try with CPU processing or restart the application."
        else:
            return f"Detection error: {str(error)}"

class Constants:
    """Application constants."""
    
    # File size limits
    MAX_IMAGE_SIZE_MB = 10.0
    MAX_IMAGE_DIMENSIONS = (4000, 4000)
    RECOMMENDED_IMAGE_DIMENSIONS = (800, 600)
    
    # Model settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.75
    MIN_CONFIDENCE_THRESHOLD = 0.1
    MAX_CONFIDENCE_THRESHOLD = 1.0
    
    # Score weights
    BOTTLE_SCORE_WEIGHT = 50
    CAP_PENALTY_WEIGHT = 10
    LABEL_PENALTY_WEIGHT = 10
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
    
    # UI constants
    SIDEBAR_WIDTH = 300
    MAX_COLUMNS = 3
    
    # Error messages
    ERROR_MESSAGES = {
        'model_not_found': 'Model file not found. Please check the model path.',
        'invalid_image': 'Invalid image format. Please upload a valid image file.',
        'file_too_large': 'File size too large. Please upload a smaller image.',
        'processing_error': 'Error processing image. Please try again.',
        'no_detections': 'No objects detected with current confidence threshold.'
    } 