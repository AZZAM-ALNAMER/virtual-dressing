"""
Skin Tone Analysis Module
Uses MediaPipe Face Detection + LAB Color Space for accurate undertone analysis
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

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
    """Analyzes skin tone and undertone from facial images using industry-standard methods"""
    
    UNDERTONE_CATEGORIES = {
        'warm': 'Warm',
        'cool': 'Cool',
        'neutral': 'Neutral'
    }
    
    def __init__(self):
        """Initialize face detection based on available MediaPipe version"""
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
        face_region = self._extract_face_region(image_rgb, detection)
        
        if face_region is None or face_region.size == 0:
            return self._analyze_without_mediapipe(image_data)
        
        # Calculate average skin color
        avg_color = self._get_average_skin_color(face_region)
        
        # Classify undertone using LAB color space
        undertone = self._classify_undertone(avg_color)
        
        return undertone
    
    def _extract_face_region(self, image_rgb: np.ndarray, detection) -> Optional[np.ndarray]:
        """
        Extract center portion of detected face region
        
        Args:
            image_rgb: RGB image
            detection: MediaPipe face detection result
            
        Returns:
            Face region as numpy array, or None if invalid
        """
        h, w, _ = image_rgb.shape
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
        
        # Validate dimensions
        if x_end <= x or y_end <= y:
            return None
        
        # Extract center portion of face (avoid hair, background, shadows)
        # Use middle 50% of face width and height
        center_margin_x = width // 4
        center_margin_y = height // 4
        
        face_center_x = x + center_margin_x
        face_center_y = y + center_margin_y
        face_center_width = width // 2
        face_center_height = height // 2
        
        # Ensure center region is valid
        if face_center_x + face_center_width > w or face_center_y + face_center_height > h:
            # Fall back to full face region
            return image_rgb[y:y_end, x:x_end]
        
        face_region = image_rgb[
            face_center_y:face_center_y + face_center_height,
            face_center_x:face_center_x + face_center_width
        ]
        
        return face_region
    
    def _analyze_without_mediapipe(self, image_data: np.ndarray) -> str:
        """
        Fallback method to analyze skin tone without face detection
        Analyzes the center region of the image
        
        Args:
            image_data: BGR image
            
        Returns:
            Undertone category
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
        Uses simple averaging since we already extracted clean face center
        
        Args:
            face_region: Face region as RGB numpy array
            
        Returns:
            Tuple of (R, G, B) average values
        """
        # Since we extracted the center portion of the face,
        # a simple average is more reliable than complex masking
        # which might exclude valid skin tones
        
        avg_color = cv2.mean(face_region)[:3]
        
        return avg_color
    
    def _classify_undertone(self, rgb_color: Tuple[float, float, float]) -> str:
        """
        Classify skin undertone using LAB color space
        Industry-standard method for accurate undertone detection
        
        LAB color space:
        - L: Lightness (0-255)
        - A: Green ↔ Red axis
        - B: Blue ↔ Yellow axis (CRITICAL for undertone)
        
        Args:
            rgb_color: Tuple of (R, G, B) values
            
        Returns:
            Undertone category: 'warm', 'cool', or 'neutral'
        """
        # Handle edge case
        if all(c == 0 for c in rgb_color):
            return 'neutral'
        
        # Convert single RGB pixel to LAB color space
        # LAB is perceptually uniform and better for color analysis
        rgb_array = np.uint8([[list(rgb_color)]])
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)[0][0]
        
        L, A, B_channel = lab
        
        # LAB B channel interpretation:
        # - 128 = neutral (no yellow or blue bias)
        # - > 128 = yellow undertones (WARM)
        # - < 128 = blue undertones (COOL)
        
        # Define threshold for neutral zone
        # Wider threshold = more people classified as neutral
        # Narrower threshold = more warm/cool classifications
        neutral_threshold = 5
        
        if B_channel > 128 + neutral_threshold:
            return 'warm'
        elif B_channel < 128 - neutral_threshold:
            return 'cool'
        else:
            return 'neutral'
    
    def get_color_compatibility_score(self, color_tone: str, skin_undertone: str) -> float:
        """
        Calculate compatibility score between clothing color and skin undertone
        
        Args:
            color_tone: Color's undertone ('warm', 'cool', 'neutral')
            skin_undertone: User's skin undertone ('warm', 'cool', 'neutral')
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Compatibility matrix based on color theory
        compatibility_matrix = {
            'warm': {
                'warm': 1.0,      # Perfect match
                'neutral': 0.8,   # Good match
                'cool': 0.5       # Poor match
            },
            'cool': {
                'cool': 1.0,
                'neutral': 0.8,
                'warm': 0.5
            },
            'neutral': {
                'warm': 0.85,
                'cool': 0.85,
                'neutral': 1.0
            }
        }
        
        return compatibility_matrix.get(skin_undertone, {}).get(color_tone, 0.5)
    
    def is_color_compatible(self, color_tone: str, skin_undertone: str, 
                          threshold: float = 0.7) -> bool:
        """
        Check if a color is compatible with skin undertone
        
        Args:
            color_tone: Color's undertone
            skin_undertone: User's undertone
            threshold: Minimum compatibility score (default: 0.7)
            
        Returns:
            True if compatible, False otherwise
        """
        score = self.get_color_compatibility_score(color_tone, skin_undertone)
        return score >= threshold
    
    def get_undertone_description(self, undertone: str) -> Dict[str, any]:
        """
        Get detailed information about an undertone
        
        Args:
            undertone: Undertone category
            
        Returns:
            Dictionary with undertone details
        """
        descriptions = {
            'warm': {
                'name': 'Warm',
                'description': 'Yellow, peachy, or golden undertones',
                'best_colors': ['warm', 'neutral'],
                'avoid_colors': ['cool'],
                'characteristics': [
                    'Veins appear greenish',
                    'Looks best in gold jewelry',
                    'Tans easily in the sun'
                ]
            },
            'cool': {
                'name': 'Cool',
                'description': 'Pink, red, or bluish undertones',
                'best_colors': ['cool', 'neutral'],
                'avoid_colors': ['warm'],
                'characteristics': [
                    'Veins appear bluish or purple',
                    'Looks best in silver jewelry',
                    'Burns easily in the sun'
                ]
            },
            'neutral': {
                'name': 'Neutral',
                'description': 'Balanced mix of warm and cool undertones',
                'best_colors': ['warm', 'cool', 'neutral'],
                'avoid_colors': [],
                'characteristics': [
                    'Veins appear blue-green',
                    'Looks good in both gold and silver jewelry',
                    'Can wear most color palettes'
                ]
            }
        }
        
        return descriptions.get(undertone, descriptions['neutral'])
    
    def analyze_with_confidence(self, image_data: np.ndarray) -> Dict[str, any]:
        """
        Analyze skin tone with confidence scoring
        
        Args:
            image_data: Image as numpy array (BGR)
            
        Returns:
            Dictionary with undertone and confidence metrics
        """
        undertone = self.analyze_skin_tone(image_data)
        
        # Get face region for confidence calculation
        if self.use_mediapipe:
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                detection = results.detections[0]
                face_region = self._extract_face_region(image_rgb, detection)
                
                if face_region is not None and face_region.size > 0:
                    # Calculate confidence based on color consistency
                    avg_color = self._get_average_skin_color(face_region)
                    std_dev = np.std(face_region, axis=(0, 1))
                    
                    # Lower std deviation = more consistent color = higher confidence
                    confidence = max(0.5, 1.0 - (np.mean(std_dev) / 255.0))
                    
                    return {
                        'undertone': undertone,
                        'confidence': round(confidence, 2),
                        'face_detected': True,
                        'description': self.get_undertone_description(undertone)
                    }
        
        # Fallback result
        return {
            'undertone': undertone,
            'confidence': 0.7,  # Default confidence for fallback method
            'face_detected': False,
            'description': self.get_undertone_description(undertone)
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection') and self.use_mediapipe:
            self.face_detection.close()


# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = SkinToneAnalyzer()
    
    # Load test image
    image = cv2.imread('test_image.jpg')
    
    if image is not None:
        # Simple analysis
        undertone = analyzer.analyze_skin_tone(image)
        print(f"Detected undertone: {undertone}")
        
        # Detailed analysis with confidence
        result = analyzer.analyze_with_confidence(image)
        print(f"\nDetailed Analysis:")
        print(f"Undertone: {result['undertone']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Face Detected: {result['face_detected']}")
        print(f"Description: {result['description']['description']}")
        
        # Test color compatibility
        color_score = analyzer.get_color_compatibility_score('warm', undertone)
        print(f"\nWarm color compatibility score: {color_score}")