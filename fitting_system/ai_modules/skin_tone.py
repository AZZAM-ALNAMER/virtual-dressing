"""
Skin Tone Analysis Module
Uses MediaPipe Face Detection to analyze skin tone
"""

import cv2
import numpy as np
from typing import Tuple, Optional

try:
    # Try new MediaPipe API (v0.10+)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except ImportError:
    # Fall back to old API
    import mediapipe as mp
    USE_NEW_API = False


class SkinToneAnalyzer:
    """Analyzes skin tone and undertone from facial images"""
    
    UNDERTONE_CATEGORIES = {
        'warm': 'Warm',
        'cool': 'Cool',
        'neutral': 'Neutral'
    }
    
    def __init__(self):
        if USE_NEW_API:
            # New API - use fallback method
            self.use_mediapipe = False
        else:
            # Old API
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
    
    def analyze_skin_tone(self, image_data: np.ndarray) -> str:
        """
        Analyze skin undertone from an image
        
        Args:
            image_data: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            Undertone category: 'warm', 'cool', or 'neutral'
        """
        if not self.use_mediapipe:
            # Fallback: Analyze center region of image
            return self._analyze_without_mediapipe(image_data)
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Detect face
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            # Fallback to center region analysis
            return self._analyze_without_mediapipe(image_data)
        
        # Get the first detected face
        detection = results.detections[0]
        
        # Extract face region
        h, w, _ = image_data.shape
        bounding_box = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        x = int(bounding_box.xmin * w)
        y = int(bounding_box.ymin * h)
        width = int(bounding_box.width * w)
        height = int(bounding_box.height * h)
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        x_end = min(w, x + width)
        y_end = min(h, y + height)
        
        # Extract face region (use center portion to avoid hair/background)
        face_center_x = x + width // 4
        face_center_y = y + height // 4
        face_center_width = width // 2
        face_center_height = height // 2
        
        face_region = image_rgb[
            face_center_y:face_center_y + face_center_height,
            face_center_x:face_center_x + face_center_width
        ]
        
        if face_region.size == 0:
            return self._analyze_without_mediapipe(image_data)
        
        # Calculate average skin color
        avg_color = self._get_average_skin_color(face_region)
        
        # Classify undertone
        undertone = self._classify_undertone(avg_color)
        
        return undertone
    
    def _analyze_without_mediapipe(self, image_data: np.ndarray) -> str:
        """
        Fallback method to analyze skin tone without face detection
        Analyzes the center region of the image
        """
        h, w, _ = image_data.shape
        
        # Extract center region (assume face/body is in center)
        center_y = h // 3
        center_x = w // 3
        region_h = h // 3
        region_w = w // 3
        
        center_region = image_data[center_y:center_y + region_h, center_x:center_x + region_w]
        
        # Convert BGR to RGB
        center_region_rgb = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
        
        # Calculate average color
        avg_color = self._get_average_skin_color(center_region_rgb)
        
        # Classify undertone
        return self._classify_undertone(avg_color)
    
    def _get_average_skin_color(self, face_region: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate average skin color from face region
        
        Args:
            face_region: Face region as RGB numpy array
            
        Returns:
            Tuple of (R, G, B) average values
        """
        # Convert to YCrCb color space for better skin detection
        ycrcb = cv2.cvtColor(face_region, cv2.COLOR_RGB2YCrCb)
        
        # Define skin color range in YCrCb
        # These values are empirically determined for skin detection
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask for skin pixels
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply mask to get skin pixels only
        skin_pixels = cv2.bitwise_and(face_region, face_region, mask=skin_mask)
        
        # Calculate average color of skin pixels
        # If no skin pixels detected, use entire face region
        if np.count_nonzero(skin_mask) > 0:
            avg_color = cv2.mean(face_region, mask=skin_mask)[:3]
        else:
            avg_color = cv2.mean(face_region)[:3]
        
        return avg_color
    
    def _classify_undertone(self, rgb_color: Tuple[float, float, float]) -> str:
        """
        Classify skin undertone based on RGB color ratios
        Determines whether undertone is warm, cool, or neutral
        
        Args:
            rgb_color: Tuple of (R, G, B) values
            
        Returns:
            Undertone category: 'warm', 'cool', or 'neutral'
        """
        r, g, b = rgb_color
        
        # Calculate ratios to determine undertone
        # Warm undertones: higher red and yellow (R+G > B)
        # Cool undertones: higher blue (B > R+G)
        # Neutral: balanced ratio
        
        if r == 0 and g == 0 and b == 0:
            return 'neutral'
        
        # Calculate color ratios
        total = r + g + b
        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total
        
        # Warm: dominant red/yellow (high R, high G relative to B)
        warm_score = (r_ratio + g_ratio) - b_ratio
        
        # Cool: dominant blue/violet (high B relative to R+G)
        cool_score = b_ratio - (r_ratio + g_ratio)
        
        # Determine undertone with threshold for neutral
        threshold = 0.05
        
        if warm_score > threshold:
            return 'warm'
        elif cool_score > threshold:
            return 'cool'
        else:
            return 'neutral'
    
    def get_recommended_colors(self, undertone: str) -> list:
        """
        Get recommended clothing colors based on skin undertone
        
        Args:
            undertone: Undertone category ('warm', 'cool', or 'neutral')
            
        Returns:
            List of recommended color names
        """
        color_recommendations = {
            'warm': [
                'Warm Coral', 'Peachy Orange', 'Terracotta', 'Burnt Orange',
                'Warm Brown', 'Golden Yellow', 'Olive Green', 'Warm Red',
                'Caramel', 'Earth Tones', 'Mustard', 'Warm Beige'
            ],
            'cool': [
                'Cool Blue', 'Icy Blue', 'Jewel Tone Purple', 'Magenta',
                'Cool Pink', 'Emerald Green', 'Plum', 'Cool Gray',
                'Navy Blue', 'Burgundy', 'Bright Red', 'Cool Mint'
            ],
            'neutral': [
                'Pure White', 'Black', 'Cool Gray', 'Warm Gray',
                'Navy Blue', 'Charcoal', 'Cream', 'True Red',
                'Forest Green', 'Deep Purple', 'Silver', 'Gold'
            ]
        }
        
        return color_recommendations.get(undertone, color_recommendations['neutral'])
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
