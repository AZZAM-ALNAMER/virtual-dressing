"""
Body Measurement Estimation Module
Uses MediaPipe Pose to estimate body measurements from images
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import urllib.request

try:
    # New MediaPipe API (v0.10+)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False


class BodyMeasurementEstimator:
    """Estimates body measurements using pose detection"""
    
    # Model file URL for MediaPipe Pose Landmarker
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    MODEL_PATH = "pose_landmarker.task"
    
    def __init__(self):
        self.use_mediapipe = False
        self.pose_landmarker = None
        
        if USE_NEW_API:
            # Download model if not exists
            if not os.path.exists(self.MODEL_PATH):
                print(f"Downloading MediaPipe Pose model...")
                try:
                    urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                    print("Model downloaded successfully!")
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    print("Using fallback measurement method")
                    return
            
            # Initialize PoseLandmarker
            try:
                base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False,
                    num_poses=1  # Detect only one person
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                self.use_mediapipe = True
                print("MediaPipe Pose initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize MediaPipe: {e}")
                print("Using fallback measurement method")
        
        # Calibration factors (pixels to cm)
        self.PIXEL_TO_CM_RATIO = 0.3  # Approximate, assumes person is ~170cm tall
    
    # Fashion segment landmark indices (MediaPipe Pose uses 33 landmarks)
    # Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    FASHION_SEGMENTS = {
        'shoulder_width': (11, 12),              # L/R Shoulder
        'torso_length': ((11, 12), (23, 24)),    # Shoulder midpoint → Hip midpoint
        'arm_length': (11, 13, 15),              # Shoulder → Elbow → Wrist
        'upper_arm': (11, 13),                   # Shoulder → Elbow
        'forearm': (13, 15),                     # Elbow → Wrist
        'inseam': (23, 25, 27),                  # Hip → Knee → Ankle
        'outseam': (23, 27),                     # Hip → Ankle (direct)
        'thigh': (23, 25),                       # Hip → Knee
        'lower_leg': (25, 27),                   # Knee → Ankle
        'hip_width': (23, 24),                   # L/R Hip
    }
    
    # Regional scale adjustments for camera perspective
    REGIONAL_SCALES = {
        'upper_body': 1.02,   # Shoulders/chest slightly closer to camera
        'torso': 1.00,        # Base reference
        'lower_body': 0.98,   # Legs slightly further from camera
    }
    
    # Fit ease adjustments (cm to add for clothing fit)
    FIT_EASE = {
        'slim': {'chest': 4, 'waist': 2, 'hip': 3},
        'regular': {'chest': 8, 'waist': 4, 'hip': 6},
        'oversize': {'chest': 14, 'waist': 8, 'hip': 10},
    }
    
    # Body-type-specific multipliers for circumference estimation (when no side image)
    CIRCUMFERENCE_MULTIPLIERS = {
        'chest': {'slim': 2.2, 'average': 2.5, 'athletic': 2.7},
        'waist': {'slim': 2.0, 'average': 2.3, 'athletic': 2.5},
        'hip': {'slim': 2.3, 'average': 2.6, 'athletic': 2.8},
    }
        
    def analyze_pose(self, image_data: np.ndarray) -> Dict:
        """Analyze pose for real-time feedback"""
        if not self.use_mediapipe or self.pose_landmarker is None:
            return {
                "detected": True, 
                "message": "System ready", 
                "status": "ready", 
                "quality": 1.0, 
                "landmarks": []
            }
            
        try:
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.pose_landmarker.detect(mp_image)
            
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return {
                    "detected": False, 
                    "message": "No person detected", 
                    "status": "bad", 
                    "quality": 0.0,
                    "landmarks": []
                }
                
            landmarks = detection_result.pose_landmarks[0]
            
            # Key landmarks
            # NOSE=0, LEFT_ANKLE=27, RIGHT_ANKLE=28
            nose = landmarks[0]
            l_ankle = landmarks[27]
            r_ankle = landmarks[28]
            
            # Check conditions
            message = "Perfect! Hold still..."
            status = "good"
            quality = 0.95
            
            # Check Feet (y goes from 0 top to 1 bottom)
            if l_ankle.y > 0.95 or r_ankle.y > 0.95:
                 message = "Feet not visible - Step Back"
                 status = "warning"
                 quality = 0.5
            # Check Head
            elif nose.y < 0.05:
                 message = "Head cut off - Adjust Camera"
                 status = "warning"
                 quality = 0.5
            else:
                 # Check if too far (height of person relative to frame)
                 person_h = ((l_ankle.y + r_ankle.y)/2) - nose.y
                 if person_h < 0.4:
                      message = "Too far - Come Closer"
                      status = "warning"
                      quality = 0.6
            
            landmarks_data = [{'x': lm.x, 'y': lm.y} for lm in landmarks]
            
            return {
                "detected": True,
                "message": message,
                "status": status,
                "quality": quality,
                "landmarks": landmarks_data
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {"detected": False, "message": "Analysis failed", "status": "error", "quality": 0.0}

    def estimate_from_image(self, image_data: np.ndarray, reference_height_cm: Optional[float] = None) -> Dict[str, float]:
        """
        Estimate body measurements from a single image
        
        Args:
            image_data: Image as numpy array (BGR format from OpenCV)
            reference_height_cm: Optional known height for calibration
            
        Returns:
            Dictionary with measurements in centimeters
        """
        if not self.use_mediapipe or self.pose_landmarker is None:
            # Fallback: Return estimated measurements based on image analysis
            return self._estimate_without_mediapipe(image_data, reference_height_cm)
        
        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect pose
            detection_result = self.pose_landmarker.detect(mp_image)
            
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                print("No pose detected, using fallback")
                return self._estimate_without_mediapipe(image_data, reference_height_cm)
            
            # Get landmarks from first detected pose
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = image_data.shape
            
            # Calculate measurements
            measurements = {}
            
            # Height: top of head to feet (used for calibration)
            height_pixels = self._calculate_height_new_api(landmarks, h, w)
            
            # Calibrate pixel-to-cm ratio if reference height is provided
            if reference_height_cm:
                self.PIXEL_TO_CM_RATIO = reference_height_cm / height_pixels
            
            # Apply regional scaling for upper body measurements
            upper_scale = self.REGIONAL_SCALES['upper_body']
            lower_scale = self.REGIONAL_SCALES['lower_body']
            
            # Height (calibration reference only, but still include)
            measurements['height'] = self.normalize_measurement(
                height_pixels * self.PIXEL_TO_CM_RATIO
            )
            
            # Shoulder width (upper body)
            shoulder_width_pixels = self._calculate_shoulder_width_new_api(landmarks, h, w)
            measurements['shoulder_width'] = self.normalize_measurement(
                shoulder_width_pixels * self.PIXEL_TO_CM_RATIO * upper_scale
            )
            
            # Chest circumference (upper body)
            chest_pixels = self._calculate_chest_new_api(landmarks, h, w)
            measurements['chest'] = self.normalize_measurement(
                chest_pixels * self.PIXEL_TO_CM_RATIO * upper_scale
            )
            
            # Waist circumference (torso)
            waist_pixels = self._calculate_waist_new_api(landmarks, h, w)
            measurements['waist'] = self.normalize_measurement(
                waist_pixels * self.PIXEL_TO_CM_RATIO
            )
            
            # Hip circumference (lower body)
            hip_pixels = self._calculate_hip_new_api(landmarks, h, w)
            measurements['hip'] = self.normalize_measurement(
                hip_pixels * self.PIXEL_TO_CM_RATIO * lower_scale
            )
            
            # Torso length (shoulder to hip)
            torso_pixels = self._calculate_torso_length_new_api(landmarks, h, w)
            measurements['torso_length'] = self.normalize_measurement(
                torso_pixels * self.PIXEL_TO_CM_RATIO
            )
            
            # Arm length (shoulder → elbow → wrist)
            arm_pixels = self._calculate_arm_length_new_api(landmarks, h, w)
            measurements['arm_length'] = self.normalize_measurement(
                arm_pixels * self.PIXEL_TO_CM_RATIO * upper_scale
            )
            
            # Inseam (hip → knee → ankle for pants)
            inseam_pixels = self._calculate_inseam_new_api(landmarks, h, w)
            measurements['inseam'] = self.normalize_measurement(
                inseam_pixels * self.PIXEL_TO_CM_RATIO * lower_scale
            )
            
            return measurements
            
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return self._estimate_without_mediapipe(image_data, reference_height_cm)
    
    def _estimate_without_mediapipe(self, image_data: np.ndarray, reference_height_cm: Optional[float] = None) -> Dict[str, float]:
        """
        Fallback method to estimate measurements without MediaPipe
        Returns average measurements based on typical body proportions
        """
        # Handle case where image_data might be None (from estimate_with_stability)
        if image_data is None:
            height = reference_height_cm if reference_height_cm else 170.0
        else:
            height = reference_height_cm if reference_height_cm else 170.0
        
        # Estimate measurements based on statistical body proportions
        # These are population averages and will be less accurate than pose detection
        measurements = {
            'height': self.normalize_measurement(height),
            'shoulder_width': self.normalize_measurement(height * 0.25),      # ~25% of height
            'chest': self.normalize_measurement(height * 0.55),               # ~55% of height
            'waist': self.normalize_measurement(height * 0.47),               # ~47% of height
            'hip': self.normalize_measurement(height * 0.55),                 # ~55% of height
            'torso_length': self.normalize_measurement(height * 0.30),        # ~30% of height
            'arm_length': self.normalize_measurement(height * 0.33),          # ~33% of height
            'inseam': self.normalize_measurement(height * 0.45),              # ~45% of height
        }
        
        return measurements
    
    def _calculate_height(self, landmarks, h: int, w: int) -> float:
        """Calculate height in pixels"""
        # Use nose (0) as top and average of ankles (27, 28) as bottom
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        top_y = nose.y * h
        bottom_y = ((left_ankle.y + right_ankle.y) / 2) * h
        
        height = abs(bottom_y - top_y)
        return height
    
    def _calculate_shoulder_width(self, landmarks, h: int, w: int) -> float:
        """Calculate shoulder width in pixels"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        left_x = left_shoulder.x * w
        right_x = right_shoulder.x * w
        left_y = left_shoulder.y * h
        right_y = right_shoulder.y * h
        
        width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        return width
    
    def _calculate_chest(self, landmarks, h: int, w: int) -> float:
        """
        Estimate chest circumference
        Approximation: chest circumference ≈ shoulder_width * 2.5
        """
        shoulder_width = self._calculate_shoulder_width(landmarks, h, w)
        # Approximate chest circumference from shoulder width
        chest_circumference = shoulder_width * 2.5
        return chest_circumference
    
    def _calculate_waist(self, landmarks, h: int, w: int) -> float:
        """
        Estimate waist circumference
        Using hip landmarks as approximation
        """
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        left_x = left_hip.x * w
        right_x = right_hip.x * w
        left_y = left_hip.y * h
        right_y = right_hip.y * h
        
        hip_width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        
        # Approximate waist circumference from hip width
        # Waist is typically slightly smaller than hips
        waist_circumference = hip_width * 2.3
        return waist_circumference
    
    def estimate_from_front_and_side(
        self, 
        front_image: np.ndarray, 
        side_image: Optional[np.ndarray] = None,
        reference_height_cm: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate measurements from front and optional side images
        
        Args:
            front_image: Front view image
            side_image: Optional side view image for better accuracy
            reference_height_cm: Optional known height for calibration
            
        Returns:
            Dictionary with measurements
        """
        # Get measurements from front image
        front_measurements = self.estimate_from_image(front_image, reference_height_cm)
        
        # If side image is provided, we could refine measurements
        # For this prototype, we'll use front image measurements
        if side_image is not None:
            try:
                side_measurements = self.estimate_from_image(side_image, reference_height_cm)
                # Average some measurements for better accuracy
                front_measurements['chest'] = round(
                    (front_measurements['chest'] + side_measurements['chest']) / 2, 2
                )
                front_measurements['waist'] = round(
                    (front_measurements['waist'] + side_measurements['waist']) / 2, 2
                )
            except Exception:
                # If side image processing fails, just use front measurements
                pass
        
        return front_measurements
    
    # New API calculation methods (for MediaPipe 0.10+)
    def _calculate_height_new_api(self, landmarks: List, h: int, w: int) -> float:
        """Calculate height in pixels using new API landmarks"""
        # Landmark indices for new API
        NOSE = 0
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        nose = landmarks[NOSE]
        left_ankle = landmarks[LEFT_ANKLE]
        right_ankle = landmarks[RIGHT_ANKLE]
        
        top_y = nose.y * h
        bottom_y = ((left_ankle.y + right_ankle.y) / 2) * h
        
        height = abs(bottom_y - top_y)
        return height
    
    def _calculate_shoulder_width_new_api(self, landmarks: List, h: int, w: int) -> float:
        """Calculate shoulder width in pixels using new API landmarks"""
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        
        left_x = left_shoulder.x * w
        right_x = right_shoulder.x * w
        left_y = left_shoulder.y * h
        right_y = right_shoulder.y * h
        
        width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        return width
    
    def _calculate_chest_new_api(self, landmarks: List, h: int, w: int, side_depth: float = None) -> float:
        """
        Estimate chest circumference using ellipse formula when side image available,
        otherwise use improved multipliers.
        """
        shoulder_width = self._calculate_shoulder_width_new_api(landmarks, h, w)
        
        if side_depth:
            # Ellipse-based estimation: C ≈ π × √(2 × (a² + b²))
            chest_circumference = self._calculate_circumference_ellipse(
                shoulder_width * self.PIXEL_TO_CM_RATIO,
                side_depth,
                'chest'
            )
            return chest_circumference / self.PIXEL_TO_CM_RATIO  # Return in pixels
        else:
            # Use average multiplier (will be converted to cm later)
            multiplier = self.CIRCUMFERENCE_MULTIPLIERS['chest']['average']
            return shoulder_width * multiplier
    
    def _calculate_waist_new_api(self, landmarks: List, h: int, w: int, side_depth: float = None) -> float:
        """
        Estimate waist circumference using ellipse formula when side image available,
        otherwise use improved multipliers.
        """
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        
        left_x = left_hip.x * w
        right_x = right_hip.x * w
        left_y = left_hip.y * h
        right_y = right_hip.y * h
        
        hip_width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        
        if side_depth:
            waist_circumference = self._calculate_circumference_ellipse(
                hip_width * self.PIXEL_TO_CM_RATIO * 0.9,  # Waist ~90% of hip width
                side_depth * 0.85,  # Waist depth ~85% of hip depth
                'waist'
            )
            return waist_circumference / self.PIXEL_TO_CM_RATIO
        else:
            multiplier = self.CIRCUMFERENCE_MULTIPLIERS['waist']['average']
            return hip_width * multiplier
    
    def _calculate_hip_new_api(self, landmarks: List, h: int, w: int, side_depth: float = None) -> float:
        """Calculate hip circumference."""
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        
        left_x = left_hip.x * w
        right_x = right_hip.x * w
        left_y = left_hip.y * h
        right_y = right_hip.y * h
        
        hip_width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        
        if side_depth:
            hip_circumference = self._calculate_circumference_ellipse(
                hip_width * self.PIXEL_TO_CM_RATIO,
                side_depth,
                'hip'
            )
            return hip_circumference / self.PIXEL_TO_CM_RATIO
        else:
            multiplier = self.CIRCUMFERENCE_MULTIPLIERS['hip']['average']
            return hip_width * multiplier
    
    def _calculate_torso_length_new_api(self, landmarks: List, h: int, w: int) -> float:
        """Calculate torso length from shoulder midpoint to hip midpoint."""
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        # Shoulder midpoint
        shoulder_mid_y = ((landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2) * h
        
        # Hip midpoint
        hip_mid_y = ((landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2) * h
        
        torso_length = abs(hip_mid_y - shoulder_mid_y)
        return torso_length
    
    def _calculate_arm_length_new_api(self, landmarks: List, h: int, w: int) -> float:
        """Calculate full arm length (shoulder → elbow → wrist)."""
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        
        shoulder = landmarks[LEFT_SHOULDER]
        elbow = landmarks[LEFT_ELBOW]
        wrist = landmarks[LEFT_WRIST]
        
        # Upper arm: shoulder to elbow
        upper_arm = np.sqrt(
            ((elbow.x - shoulder.x) * w)**2 + 
            ((elbow.y - shoulder.y) * h)**2
        )
        
        # Forearm: elbow to wrist
        forearm = np.sqrt(
            ((wrist.x - elbow.x) * w)**2 + 
            ((wrist.y - elbow.y) * h)**2
        )
        
        return upper_arm + forearm
    
    def _calculate_inseam_new_api(self, landmarks: List, h: int, w: int) -> float:
        """Calculate inseam (hip → knee → ankle for pants length)."""
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        
        hip = landmarks[LEFT_HIP]
        knee = landmarks[LEFT_KNEE]
        ankle = landmarks[LEFT_ANKLE]
        
        # Upper leg: hip to knee
        upper_leg = np.sqrt(
            ((knee.x - hip.x) * w)**2 + 
            ((knee.y - hip.y) * h)**2
        )
        
        # Lower leg: knee to ankle
        lower_leg = np.sqrt(
            ((ankle.x - knee.x) * w)**2 + 
            ((ankle.y - knee.y) * h)**2
        )
        
        return upper_leg + lower_leg
    
    def _calculate_circumference_ellipse(self, front_width: float, side_depth: float, body_part: str = 'chest') -> float:
        """
        Fashion-grade circumference estimation using ellipse formula.
        C ≈ π × √(2 × (a² + b²)) where a, b are semi-axes
        
        Args:
            front_width: Width from front view in cm
            side_depth: Depth from side view in cm
            body_part: 'chest', 'waist', or 'hip'
            
        Returns:
            Estimated circumference in cm
        """
        a = front_width / 2  # Semi-major axis
        b = side_depth / 2   # Semi-minor axis
        
        # Ramanujan's approximation for ellipse circumference
        circumference = np.pi * np.sqrt(2 * (a**2 + b**2))
        
        return circumference
    
    @staticmethod
    def normalize_measurement(value: float, round_to: float = 0.5, min_val: float = 0, max_val: float = 300) -> float:
        """
        Fashion-grade normalization: round to nearest increment and clamp to valid range.
        
        Args:
            value: Raw measurement value in cm
            round_to: Round to nearest value (default 0.5 cm)
            min_val: Minimum valid value
            max_val: Maximum valid value
            
        Returns:
            Normalized measurement
        """
        # Clamp to valid range
        clamped = max(min_val, min(max_val, value))
        # Round to nearest increment
        return round(clamped / round_to) * round_to
    
    def apply_fit_ease(self, measurements: Dict[str, float], fit_type: str = 'regular') -> Dict[str, float]:
        """
        Add ease allowance for clothing fit.
        
        Args:
            measurements: Raw body measurements
            fit_type: 'slim', 'regular', or 'oversize'
            
        Returns:
            Measurements with ease added for clothing
        """
        ease = self.FIT_EASE.get(fit_type, self.FIT_EASE['regular'])
        result = measurements.copy()
        
        for key, ease_value in ease.items():
            if key in result:
                result[key] = result[key] + ease_value
        
        return result
    
    def estimate_with_stability(
        self, 
        frames: List[np.ndarray], 
        reference_height_cm: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate measurements from multiple frames for stability.
        Uses median for noise rejection.
        
        Args:
            frames: List of image frames (BGR format)
            reference_height_cm: Optional known height for calibration
            
        Returns:
            Dictionary with stable measurements
        """
        if not frames:
            return self._estimate_without_mediapipe(None, reference_height_cm)
        
        all_measurements = []
        
        for frame in frames:
            try:
                m = self.estimate_from_image(frame, reference_height_cm)
                if m:
                    all_measurements.append(m)
            except Exception:
                continue
        
        if not all_measurements:
            return self._estimate_without_mediapipe(frames[0], reference_height_cm)
        
        # Use median for each measurement to reject outliers
        stable_measurements = {}
        keys = all_measurements[0].keys()
        
        for key in keys:
            values = [m[key] for m in all_measurements if key in m]
            if values:
                stable_measurements[key] = self.normalize_measurement(np.median(values))
        
        return stable_measurements
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
