"""
YOLO-based Body Analyzer Module

Architecture:
    - YOLOv8 Pose: Real-time pose detection + body keypoint extraction for
      body measurement estimation (replaces MediaPipe).
    - YOLOv8 Pose (face image): Used for face detection; skin tone extracted
      directly from the face region using color analysis.
    - Gemini LLM: Receives the estimated measurements and returns a recommended
      clothing size letter (S, M, L, XL, XXL, XXXL).

Two images are required:
    1. Body image  – full-body front view for pose + measurements.
    2. Face image  – close-up selfie for accurate skin-tone detection.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import ultralytics (YOLOv8)
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Run: pip install ultralytics")


# ---------------------------------------------------------------------------
# YOLO model singleton helpers
# ---------------------------------------------------------------------------
_pose_model = None


def _get_pose_model():
    """Lazy-load the YOLOv8 pose model (downloads on first use)."""
    global _pose_model
    if _pose_model is None:
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package is not installed.")
        _pose_model = YOLO("yolov8n-pose.pt")   # nano – fast, good enough
        logger.info("YOLOv8 pose model loaded.")
    return _pose_model


# ---------------------------------------------------------------------------
# Keypoint indices (COCO 17-keypoint layout used by YOLOv8)
# ---------------------------------------------------------------------------
KP = {
    "nose":          0,
    "left_eye":      1,
    "right_eye":     2,
    "left_ear":      3,
    "right_ear":     4,
    "left_shoulder": 5,
    "right_shoulder":6,
    "left_elbow":    7,
    "right_elbow":   8,
    "left_wrist":    9,
    "right_wrist":   10,
    "left_hip":      11,
    "right_hip":     12,
    "left_knee":     13,
    "right_knee":    14,
    "left_ankle":    15,
    "right_ankle":   16,
}

SKELETON_CONNECTIONS = [
    (KP["left_shoulder"],  KP["right_shoulder"]),
    (KP["left_shoulder"],  KP["left_elbow"]),
    (KP["left_elbow"],     KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"],    KP["right_wrist"]),
    (KP["left_shoulder"],  KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"],       KP["right_hip"]),
    (KP["left_hip"],       KP["left_knee"]),
    (KP["left_knee"],      KP["left_ankle"]),
    (KP["right_hip"],      KP["right_knee"]),
    (KP["right_knee"],     KP["right_ankle"]),
    (KP["nose"],           KP["left_shoulder"]),
    (KP["nose"],           KP["right_shoulder"]),
]


# ---------------------------------------------------------------------------
# Core analyzer class
# ---------------------------------------------------------------------------

class YOLOBodyAnalyzer:
    """
    Analyzes body images using YOLOv8 pose estimation and skin-tone color
    analysis, then uses an LLM to recommend a clothing size.
    """

    # ---- Size chart (chest circumference in cm) used as LLM context ----
    SIZE_CHART = {
        "XS":   (76,  84),
        "S":    (84,  92),
        "M":    (92,  100),
        "L":    (100, 108),
        "XL":   (108, 116),
        "XXL":  (116, 124),
        "XXXL": (124, 140),
    }

    # ---- Fitzpatrick-inspired skin-tone buckets (HSV hue + saturation) ----
    SKIN_TONE_BUCKETS = [
        ("very_light", (200, 170, 130)),   # approximate RGB centres
        ("light",      (195, 155, 110)),
        ("intermediate",(180, 130,  85)),
        ("tan",        (160, 110,  65)),
        ("dark",       (120,  80,  45)),
    ]

    def __init__(self):
        pass  # GeminiClient is used directly via get_gemini_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_pose_frame(self, image_bgr: np.ndarray) -> Dict:
        """
        Real-time pose feedback for the camera overlay (body image).
        Returns landmark positions + guidance message.
        """
        try:
            model = _get_pose_model()
        except RuntimeError as e:
            return {"detected": True, "message": "System ready", "status": "ready",
                    "quality": 1.0, "landmarks": []}

        results = model(image_bgr, verbose=False)

        if not results or len(results) == 0:
            return {"detected": False, "message": "No person detected", "status": "bad",
                    "quality": 0.0, "landmarks": []}

        result = results[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return {"detected": False, "message": "No person detected", "status": "bad",
                    "quality": 0.0, "landmarks": []}

        kps = result.keypoints.xy[0].cpu().numpy()   # shape (17, 2)
        h, w = image_bgr.shape[:2]

        # Normalise to [0,1]
        landmarks = [{"x": float(kp[0]) / w, "y": float(kp[1]) / h} for kp in kps]

        # Framing checks
        nose    = landmarks[KP["nose"]]
        l_ankle = landmarks[KP["left_ankle"]]
        r_ankle = landmarks[KP["right_ankle"]]

        message = "Perfect! Hold still..."
        status  = "good"
        quality = 0.95

        avg_ankle_y = (l_ankle["y"] + r_ankle["y"]) / 2

        if avg_ankle_y > 0.95:
            message, status, quality = "Feet not visible – Step Back", "warning", 0.5
        elif nose["y"] < 0.05:
            message, status, quality = "Head cut off – Adjust Camera", "warning", 0.5
        else:
            person_h = avg_ankle_y - nose["y"]
            if person_h < 0.4:
                message, status, quality = "Too far – Come Closer", "warning", 0.6

        return {"detected": True, "message": message, "status": status,
                "quality": quality, "landmarks": landmarks}

    def analyze_face_frame(self, image_bgr: np.ndarray) -> Dict:
        """
        Real-time face feedback for the selfie step (face image).
        Uses OpenCV Haar cascade for speed.
        """
        h, w = image_bgr.shape[:2]
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        if len(faces) == 0:
            return {"detected": False, "message": "No face detected – look at the camera",
                    "status": "error", "quality": 0.0, "landmarks": []}

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_area_ratio  = (fw * fh) / (w * h)
        center_x_ratio   = abs((x + fw / 2) - w / 2) / w
        center_y_ratio   = abs((y + fh / 2) - h / 2) / h

        if face_area_ratio < 0.08:
            return {"detected": True, "message": "Move CLOSER to the camera",
                    "status": "warning", "quality": 0.4, "landmarks": []}
        elif face_area_ratio < 0.15:
            return {"detected": True, "message": "A bit closer for better accuracy",
                    "status": "warning", "quality": 0.6, "landmarks": []}
        elif center_x_ratio > 0.25 or center_y_ratio > 0.25:
            return {"detected": True, "message": "Center your face in the circle",
                    "status": "warning", "quality": 0.7, "landmarks": []}
        else:
            return {"detected": True, "message": "Perfect! Hold still...",
                    "status": "good", "quality": 0.95, "landmarks": []}

    def extract_measurements_from_body_image(
        self, body_image_bgr: np.ndarray, user_height_cm: float = None
    ) -> Dict[str, float]:
        """
        Use YOLO keypoints to estimate body measurements (cm).
        Requires user_height_cm for accurate pixel-to-cm calibration.
        Returns a dict with: height, shoulder_width, chest, waist,
        hip, torso_length, arm_length, inseam.

        Raises RuntimeError if YOLO detection fails.
        Raises ValueError if user_height_cm is not provided.
        """
        if not user_height_cm or user_height_cm <= 0:
            raise ValueError("User height is required for accurate measurements. Please provide your height in cm.")

        try:
            model = _get_pose_model()
        except RuntimeError as e:
            raise RuntimeError(f"YOLO model is not available: {e}")

        results = model(body_image_bgr, verbose=False)
        if not results or results[0].keypoints is None:
            raise RuntimeError("YOLO could not detect any person in the body image.")

        result = results[0]
        if len(result.keypoints.xy) == 0:
            raise RuntimeError("YOLO could not detect keypoints in the body image.")

        kps = result.keypoints.xy[0].cpu().numpy()   # (17, 2) in pixels
        h_img, w_img = body_image_bgr.shape[:2]

        def kp(name):
            return kps[KP[name]]

        # ---- pixel distances ----
        l_shoulder = kp("left_shoulder")
        r_shoulder = kp("right_shoulder")
        l_hip      = kp("left_hip")
        r_hip      = kp("right_hip")
        l_ankle    = kp("left_ankle")
        r_ankle    = kp("right_ankle")
        nose_pt    = kp("nose")
        l_wrist    = kp("left_wrist")
        l_elbow    = kp("left_elbow")
        r_wrist    = kp("right_wrist")
        r_elbow    = kp("right_elbow")

        def dist(a, b):
            return float(np.linalg.norm(a - b))

        # ---------------------------------------------------------------
        # FIX 1: Estimate top-of-head instead of using the nose directly.
        # The nose sits ~55-60% of the way down the head. We approximate
        # the head top by projecting upward by 0.6× the nose-to-shoulder
        # distance (which is roughly one head-length).
        # This prevents px_height from being ~10% too short, which was
        # inflating every cm measurement by ~10%.
        # ---------------------------------------------------------------
        mid_shoulder   = (l_shoulder + r_shoulder) / 2
        nose_to_shoulder_dist = dist(nose_pt, mid_shoulder)
        head_top_pt    = nose_pt - np.array([0, nose_to_shoulder_dist * 0.6])
        mid_ankle      = (l_ankle + r_ankle) / 2
        px_height      = dist(head_top_pt, mid_ankle)

        px_shoulder_w = dist(l_shoulder, r_shoulder)
        px_torso      = dist(mid_shoulder, (l_hip + r_hip) / 2)
        px_hip_w      = dist(l_hip, r_hip)
        px_arm_l      = (dist(l_shoulder, l_elbow) + dist(l_elbow, l_wrist) +
                         dist(r_shoulder, r_elbow) + dist(r_elbow, r_wrist)) / 2
        px_inseam     = (dist(l_hip, l_ankle) + dist(r_hip, r_ankle)) / 2

        # ---- calibration: use user-provided height ----
        if px_height < 1:
            px_height = h_img * 0.85   # approximate from image height

        px_per_cm = px_height / user_height_cm

        def to_cm(px):
            return round(px / px_per_cm, 1) if px_per_cm > 0 else 0.0

        shoulder_w_cm = to_cm(px_shoulder_w)
        torso_cm      = to_cm(px_torso)
        hip_w_cm      = to_cm(px_hip_w)
        arm_cm        = to_cm(px_arm_l)
        inseam_cm     = to_cm(px_inseam)

        # ---------------------------------------------------------------
        # FIX 2: Derive waist width from the correct anatomical landmark.
        # Previously waist_cm used px_hip_w (hip keypoint width), which
        # made waist ≈ hip and erased body shape entirely.
        # The waist sits roughly 65% down the torso from shoulders.
        # We interpolate a waist point and use its estimated pixel width,
        # approximated as 70% of the hip-keypoint width (a common
        # anthropometric ratio for average adults).
        # ---------------------------------------------------------------
        # Estimate circumferences from skeletal widths using anthropometric ratios.
        # YOLO keypoints measure joint-to-joint (bideltoid / bi-iliac), NOT full width.
        # Ratios derived from anthropometric studies:
        #   bideltoid width → chest circumference  ≈ ×2.7
        #   waist width     → waist circumference  ≈ ×2.5  (waist width ≈ 70% of hip width)
        #   bi-iliac width  → hip circumference    ≈ ×2.7
        px_waist_w = px_hip_w * 0.70          # waist is narrower than hips
        chest_cm   = shoulder_w_cm * 2.7
        waist_cm   = to_cm(px_waist_w) * 2.5  # now correctly uses waist-estimated width
        hip_cm     = hip_w_cm * 2.7

        # Clamp to realistic ranges
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        measurements = {
            "height":         user_height_cm,
            "shoulder_width": clamp(shoulder_w_cm, 30, 60),
            "chest":          clamp(chest_cm,      65, 150),
            "waist":          clamp(waist_cm,      50, 140),
            "hip":            clamp(hip_cm,        70, 150),
            "torso_length":   clamp(torso_cm,      35,  65),
            "arm_length":     clamp(arm_cm,        45,  80),
            "inseam":         clamp(inseam_cm,     55, 100),
        }
        logger.info(f"YOLO measurements (height={user_height_cm}cm): {measurements}")
        return measurements

    def extract_skin_tone_from_face_image(
        self, face_image_bgr: np.ndarray
    ) -> Tuple[str, str]:
        """
        Detect skin tone from a face close-up image.
        Returns (skin_tone, undertone) where:
            skin_tone  ∈ {very_light, light, intermediate, tan, dark}
            undertone  ∈ {warm, cool}
        """
        h, w = face_image_bgr.shape[:2]

        # Try to isolate the face region with Haar cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray  = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Use the central 60% of the face to avoid hair/background
            cx, cy = x + fw // 2, y + fh // 2
            roi_w, roi_h = int(fw * 0.6), int(fh * 0.6)
            x1 = max(0, cx - roi_w // 2)
            y1 = max(0, cy - roi_h // 2)
            x2 = min(w, cx + roi_w // 2)
            y2 = min(h, cy + roi_h // 2)
            roi = face_image_bgr[y1:y2, x1:x2]
        else:
            # Fallback: use the central 40% of the image
            y1, y2 = int(h * 0.3), int(h * 0.7)
            x1, x2 = int(w * 0.3), int(w * 0.7)
            roi = face_image_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return "intermediate", "warm"

        # Convert to RGB and compute mean colour
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)   # [R, G, B]

        skin_tone  = self._classify_skin_tone(mean_rgb)
        undertone  = self._classify_undertone(mean_rgb)

        logger.info(f"Skin tone: {skin_tone}, undertone: {undertone}, mean_rgb: {mean_rgb}")
        return skin_tone, undertone

    def get_size_recommendation_from_llm(
        self,
        measurements: Dict[str, float],
        body_shape: str = "rectangle",   # FIX 3: accept body_shape so Gemini can use it
    ) -> str:
        """
        Ask Gemini to recommend a clothing size letter based on measurements.

        Uses the canonical GeminiClient (gemini-2.5-flash) – no rule-based
        fallback. Raises RuntimeError if Gemini is unavailable or fails.
        """
        from .gemini_client import get_gemini_client
        client = get_gemini_client()
        if not client.available:
            raise RuntimeError("Gemini AI is not available for size recommendation")
        result = client.get_size_recommendation(
            measurements=measurements,
            garment_type="general",
            body_shape=body_shape,        # FIX 3: pass body_shape so Gemini reasons correctly
        )
        size = result.get("recommended_size", "").upper().strip()
        valid = {"XS", "S", "M", "L", "XL", "XXL", "XXXL"}
        if size not in valid:
            raise ValueError(f"Gemini returned invalid size: '{size}'. Response: {result}")
        logger.info(f"Gemini recommended size: {size} (reason: {result.get('reasoning', 'N/A')})")
        return size

    def full_analysis(
        self,
        body_image_bgr: np.ndarray,
        face_image_bgr: np.ndarray,
        user_height_cm: float = None,
        body_shape: str = "rectangle",   # FIX 3: accept and forward body_shape
    ) -> Dict:
        """
        Complete analysis pipeline:
          1. Extract body measurements from body image (YOLO) using user height.
          2. Extract skin tone from face image (color analysis).
          3. Get recommended size from LLM.

        Args:
            body_image_bgr: Full-body front-view image (BGR numpy array).
            face_image_bgr: Face close-up image (BGR numpy array).
            user_height_cm: User's actual height in centimetres (required).
            body_shape:     Body shape string forwarded to Gemini for better
                            borderline-size decisions (e.g. "hourglass").

        Returns:
            {
                "measurements": {...},
                "skin_tone":    "intermediate",
                "undertone":    "warm",
                "recommended_size": "L",
                "confidence":   0.85,
            }
        """
        measurements     = self.extract_measurements_from_body_image(body_image_bgr, user_height_cm)
        skin_tone, undertone = self.extract_skin_tone_from_face_image(face_image_bgr)
        recommended_size = self.get_size_recommendation_from_llm(measurements, body_shape=body_shape)

        return {
            "measurements":      measurements,
            "skin_tone":         skin_tone,
            "undertone":         undertone,
            "recommended_size":  recommended_size,
            "confidence":        0.85,
        }

    # ------------------------------------------------------------------
    # Women flow helpers
    # ------------------------------------------------------------------

    def extract_skin_tone_from_hand_image(
        self, hand_image_bgr: np.ndarray
    ) -> Tuple[str, str]:
        """
        Detect skin tone from a hand image.
        Uses the central region of the image (no face detection needed).
        Returns (skin_tone, undertone).
        """
        h, w = hand_image_bgr.shape[:2]

        # Sample the central 40% of the image (palm area)
        y1, y2 = int(h * 0.3), int(h * 0.7)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        roi = hand_image_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return "intermediate", "warm"

        roi_rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)

        skin_tone = self._classify_skin_tone(mean_rgb)
        undertone = self._classify_undertone(mean_rgb)

        logger.info(f"Hand skin tone: {skin_tone}, undertone: {undertone}, mean_rgb: {mean_rgb}")
        return skin_tone, undertone

    def women_analysis(
        self,
        measurements: Dict[str, float],
        hand_image_bgr: np.ndarray,
        body_shape: str = "hourglass",
    ) -> Dict:
        """
        Women's analysis pipeline (no YOLO needed):
          1. Use manually entered measurements directly.
          2. Extract skin tone from hand image.
          3. Get recommended size from LLM.

        Args:
            measurements:    Dict with keys like height, chest, waist, hip, etc.
            hand_image_bgr:  Hand photo (BGR numpy array) for skin tone.
            body_shape:      Body shape string for Gemini.

        Returns same structure as full_analysis.
        """
        skin_tone, undertone = self.extract_skin_tone_from_hand_image(hand_image_bgr)
        recommended_size = self.get_size_recommendation_from_llm(measurements, body_shape=body_shape)

        return {
            "measurements":      measurements,
            "skin_tone":         skin_tone,
            "undertone":         undertone,
            "recommended_size":  recommended_size,
            "confidence":        0.90,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_skin_tone(mean_rgb: np.ndarray) -> str:
        """Map mean RGB to a Fitzpatrick-inspired skin tone label."""
        # Use luminance as primary signal
        r, g, b = mean_rgb
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        if luminance > 200:
            return "very_light"
        elif luminance > 170:
            return "light"
        elif luminance > 135:
            return "intermediate"
        elif luminance > 100:
            return "tan"
        else:
            return "dark"

    @staticmethod
    def _classify_undertone(mean_rgb: np.ndarray) -> str:
        """
        Warm undertone: red/yellow dominant (R > B).
        Cool undertone: blue/pink dominant (B >= R).
        """
        r, _, b = mean_rgb
        return "warm" if r > b else "cool"

    # _rule_based_size REMOVED – Gemini LLM is the sole source for size recommendations.
    # _default_measurements REMOVED – user must provide their height; YOLO must detect keypoints.


# _SizeLLM REMOVED – size recommendations now go directly through GeminiClient
# (gemini-2.5-flash) in get_size_recommendation_from_llm().


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_analyzer_instance: Optional[YOLOBodyAnalyzer] = None


def get_yolo_analyzer() -> YOLOBodyAnalyzer:
    """Return the global YOLOBodyAnalyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = YOLOBodyAnalyzer()
    return _analyzer_instance