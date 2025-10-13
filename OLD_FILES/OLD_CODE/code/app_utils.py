"""
Application utility functions and constants for the PET Detection System.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

class ColorAnalyzer:
    """Color analysis and segmentation utilities."""
    
    @staticmethod
    def extract_dominant_colors(image: np.ndarray, n_colors: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image using K-means clustering."""
        try:
            # Reshape image to 2D array
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get colors and their counts
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            color_counts = Counter(labels)
            
            # Sort by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_colors = [colors[color_id] for color_id, _ in sorted_colors]
            
            return dominant_colors
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []
    
    @staticmethod
    def is_transparent_by_dominant_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Determine if a bottle is transparent by checking if the dominant color is in the white to dark gray range.
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        dominant_colors = ColorAnalyzer.extract_dominant_colors(roi, n_colors=1)
        if not dominant_colors:
            return False
        r, g, b = dominant_colors[0]
        # ขวดใส: สีหลักอยู่ในช่วงขาวถึงเทาเข้ม (r,g,b > 100 ทุก channel)
        if r > 100 and g > 100 and b > 100:
            return True
        return False
    
    @staticmethod
    def get_bottle_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Analyze bottle color and return color information."""
        try:
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            
            # Extract dominant colors
            dominant_colors = ColorAnalyzer.extract_dominant_colors(roi, n_colors=3)
            
            if not dominant_colors:
                return {"color_name": "Unknown", "rgb": (0, 0, 0), "is_transparent": False}
            
            # Get the most dominant color
            main_color = dominant_colors[0]
            r, g, b = main_color
            
            # Determine color name
            color_name = ColorAnalyzer._get_color_name(r, g, b)
            
            # ใช้ dominant color เป็นหลักในการตัดสินขวดใส
            is_transparent = ColorAnalyzer.is_transparent_by_dominant_color(image, bbox)
            
            return {
                "color_name": color_name,
                "rgb": (r, g, b),
                "is_transparent": is_transparent,
                "transparent_percentage": 100.0 if is_transparent else 0.0,
                "color_variance": 0.0,
                "dominant_colors": dominant_colors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bottle color: {e}")
            return {"color_name": "Unknown", "rgb": (0, 0, 0), "is_transparent": False}
    
    @staticmethod
    def _get_color_name(r: int, g: int, b: int) -> str:
        """Convert RGB values to color name."""
        # Define color ranges
        color_ranges = {
            "ขาว": [(200, 200, 200), (255, 255, 255)],
            "ดำ": [(0, 0, 0), (50, 50, 50)],
            "แดง": [(150, 0, 0), (255, 100, 100)],
            "เขียว": [(0, 150, 0), (100, 255, 100)],
            "น้ำเงิน": [(0, 0, 150), (100, 100, 255)],
            "เหลือง": [(200, 200, 0), (255, 255, 100)],
            "ส้ม": [(200, 100, 0), (255, 150, 100)],
            "ม่วง": [(100, 0, 100), (200, 100, 200)],
            "ชมพู": [(200, 100, 150), (255, 150, 200)],
            "น้ำตาล": [(100, 50, 0), (150, 100, 50)],
            "เทา": [(100, 100, 100), (150, 150, 150)]
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            if all(l <= c <= u for c, (l, u) in zip((r, g, b), zip(lower, upper))):
                return color_name
        
        return "อื่นๆ"

class ObjectMeasurement:
    """Object measurement and size analysis utilities."""
    
    def __init__(self, reference_object_width_mm: float = 100.0):
        """Initialize with reference object width in mm."""
        self.reference_object_width_mm = reference_object_width_mm
        self.pixels_per_mm = None
    
    def calibrate_from_reference(self, reference_pixel_width: float):
        """Calibrate measurement using reference object."""
        self.pixels_per_mm = reference_pixel_width / self.reference_object_width_mm
        logger.info(f"Calibrated: {self.pixels_per_mm:.2f} pixels/mm")
    
    def measure_bottle_dimensions(self, bbox: Tuple[int, int, int, int], 
                                 image_shape: Tuple[int, int, int]) -> Dict:
        """Measure bottle dimensions from bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Calculate pixel dimensions
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            
            # Calculate area in pixels
            area_pixels = width_pixels * height_pixels
            
            # Convert to real-world measurements if calibrated
            if self.pixels_per_mm is not None:
                width_mm = width_pixels / self.pixels_per_mm
                height_mm = height_pixels / self.pixels_per_mm
                area_mm2 = area_pixels / (self.pixels_per_mm ** 2)
                
                # Estimate bottle volume (assuming cylindrical shape)
                # Volume = π * (diameter/2)² * height
                diameter_mm = width_mm  # Assuming width is diameter
                volume_ml = (np.pi * (diameter_mm/2)**2 * height_mm) / 1000  # Convert to ml
                
                # Determine bottle size category
                size_category = ObjectMeasurement._get_bottle_size_category(volume_ml)
                
                return {
                    "width_pixels": width_pixels,
                    "height_pixels": height_pixels,
                    "area_pixels": area_pixels,
                    "width_mm": round(width_mm, 2),
                    "height_mm": round(height_mm, 2),
                    "area_mm2": round(area_mm2, 2),
                    "volume_ml": round(volume_ml, 2),
                    "size_category": size_category,
                    "calibrated": True
                }
            else:
                # Return pixel measurements only
                return {
                    "width_pixels": width_pixels,
                    "height_pixels": height_pixels,
                    "area_pixels": area_pixels,
                    "calibrated": False
                }
                
        except Exception as e:
            logger.error(f"Error measuring bottle dimensions: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_bottle_size_category(volume_ml: float) -> str:
        """Categorize bottle by volume."""
        if volume_ml < 100:
            return "เล็กมาก (<100ml)"
        elif volume_ml < 300:
            return "เล็ก (100-300ml)"
        elif volume_ml < 600:
            return "กลาง (300-600ml)"
        elif volume_ml < 1000:
            return "ใหญ่ (600-1000ml)"
        else:
            return "ใหญ่มาก (>1000ml)"
    
    def estimate_bottle_capacity(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate bottle capacity in ml."""
        try:
            measurements = self.measure_bottle_dimensions(bbox, (0, 0, 0))
            if "volume_ml" in measurements:
                return measurements["volume_ml"]
            return 0.0
        except Exception as e:
            logger.error(f"Error estimating bottle capacity: {e}")
            return 0.0

class BottleAnalyzer:
    """Comprehensive bottle analysis combining color and measurement."""
    
    def __init__(self, reference_object_width_mm: float = 100.0):
        """Initialize bottle analyzer."""
        self.measurement = ObjectMeasurement(reference_object_width_mm)
    
    def analyze_bottle(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """Perform comprehensive bottle analysis."""
        try:
            # Color analysis
            color_info = ColorAnalyzer.get_bottle_color(image, bbox)
            
            # Size measurement
            size_info = self.measurement.measure_bottle_dimensions(bbox, image.shape)
            
            # Combine results
            analysis = {
                "color": color_info,
                "size": size_info,
                "bbox": bbox,
                "analysis_timestamp": np.datetime64('now')
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing bottle: {e}")
            return {"error": str(e)}
    
    def classify_bottle_type(self, analysis: Dict) -> str:
        """Classify bottle type based on color and size."""
        try:
            color_name = analysis.get("color", {}).get("color_name", "Unknown")
            is_transparent = analysis.get("color", {}).get("is_transparent", False)
            size_category = analysis.get("size", {}).get("size_category", "Unknown")
            
            if is_transparent:
                return f"ขวดใส {size_category}"
            else:
                return f"ขวดสี{color_name} {size_category}"
                
        except Exception as e:
            logger.error(f"Error classifying bottle type: {e}")
            return "Unknown"
    
    def get_bottle_summary(self, analysis: Dict) -> Dict:
        """Get summary of bottle analysis."""
        try:
            bottle_type = self.classify_bottle_type(analysis)
            color_info = analysis.get("color", {})
            size_info = analysis.get("size", {})
            
            summary = {
                "type": bottle_type,
                "color": color_info.get("color_name", "Unknown"),
                "is_transparent": color_info.get("is_transparent", False),
                "width_mm": size_info.get("width_mm", 0),
                "height_mm": size_info.get("height_mm", 0),
                "volume_ml": size_info.get("volume_ml", 0),
                "size_category": size_info.get("size_category", "Unknown")
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting bottle summary: {e}")
            return {"error": str(e)}

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