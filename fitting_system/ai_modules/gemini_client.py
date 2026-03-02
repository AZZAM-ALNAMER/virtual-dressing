"""
Gemini AI Client Module
Central AI engine for body measurements and recommendations.
"""

import json
import base64
import logging
import re
from typing import Dict, List, Optional, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai package not installed. Run: pip install google-generativeai")


class GeminiClient:
    """
    Wrapper around Google Gemini API for fashion AI tasks.
    """
    
    # gemini-2.5-flash: fastest valid model as of Feb 2026
    MODEL_NAME = "gemini-2.5-flash"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)
        self.model = None
        self.available = False
        
        if not GEMINI_AVAILABLE:
            return
            
        if not self.api_key:
            logger.error("No Gemini API key configured.")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            # Using system_instruction to set the "persona" permanently for the model
            self.model = genai.GenerativeModel(
                model_name=self.MODEL_NAME,
                system_instruction="You are an expert anthropometric AI. Your goal is to visually analyze human body proportions and provide highly accurate clothing size and measurement data."
            )
            self.available = True
            logger.info(f"Gemini AI client ({self.MODEL_NAME}) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    def _encode_image_for_gemini(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("utf-8")
        }
    
    def _parse_json_response(self, text: str) -> dict:
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        json_obj_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_obj_match:
            text = json_obj_match.group(1)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}

    def analyze_body(
        self,
        front_image_bytes: bytes,
        side_image_bytes: bytes = None,
        reference_height_cm: float = None
    ) -> Dict:
        if not self.available:
            raise RuntimeError("Gemini AI service is not available")
        
        height_info = f"The person's actual height is {reference_height_cm} cm." if reference_height_cm else "Estimate height based on surroundings."
        
        prompt = f"""Analyze the person in the image(s) for a virtual fitting room.
        {height_info}

        1. Extract precise body measurements in cm.
        2. Identify body shape (hourglass, rectangle, triangle, inverted_triangle, oval).
        3. Identify skin tone (light, medium, dark) and undertone (warm, cool).

        Return ONLY a JSON object:
        {{
            "measurements": {{
                "height": <cm>,
                "shoulder_width": <cm>,
                "chest": <cm>,
                "waist": <cm>,
                "hip": <cm>,
                "torso_length": <cm>,
                "arm_length": <cm>,
                "inseam": <cm>
            }},
            "body_shape": "...",
            "skin_tone": "...",
            "undertone": "...",
            "confidence": <0.0-1.0>
        }}"""

        content_parts = [prompt, self._encode_image_for_gemini(front_image_bytes)]
        if side_image_bytes:
            content_parts.append(self._encode_image_for_gemini(side_image_bytes))
        
        response = self.model.generate_content(content_parts)
        result = self._parse_json_response(response.text)
        
        if not result or "measurements" not in result:
            raise ValueError("Incomplete data from Gemini")
            
        result["measurements"] = self._validate_measurements(result["measurements"], reference_height_cm)
        return result

    def _validate_measurements(self, raw: Dict, reference_height: float = None) -> Dict[str, float]:
        """Validates measurements without 'flattening' the person's unique shape."""
        validated = {}
        # Get height first as it is our primary scale
        height = float(raw.get("height", reference_height or 175))
        if reference_height: height = reference_height

        # Dynamic ratios: We allow a range rather than a fixed multiplier
        # This prevents an athletic person from being 'clamped' into a rectangle shape
        proportions = {
            "shoulder_width": (0.20, 0.35), # 20% to 35% of height
            "chest": (0.45, 0.75),
            "waist": (0.35, 0.70),
            "hip": (0.45, 0.75),
            "torso_length": (0.25, 0.35),
            "arm_length": (0.30, 0.40),
            "inseam": (0.40, 0.50),
        }

        validated["height"] = height
        for key, (min_ratio, max_ratio) in proportions.items():
            val = float(raw.get(key, height * ((min_ratio + max_ratio) / 2)))
            
            # Check if value is within a 'realistic' human ratio
            lower_bound = height * min_ratio
            upper_bound = height * max_ratio
            
            # Clamp to the bounds to avoid 'impossible' measurements
            final_val = max(lower_bound, min(upper_bound, val))
            validated[key] = round(final_val * 2) / 2 # Round to nearest 0.5cm

        return validated

    def get_size_recommendation(self, measurements: Dict, garment_type: str, body_shape: str = "rectangle", available_sizes: List[str] = None) -> Dict:
        if not self.available:
            raise RuntimeError("Gemini AI is not available for size recommendation")

        if available_sizes is None:
            available_sizes = ["XS", "S", "M", "L", "XL", "XXL"]

        prompt = f"""You are an expert fashion sizing specialist.

Analyze the following body measurements and recommend the single best clothing size.

Body Measurements (all values in centimetres):
{json.dumps(measurements, indent=2)}

Garment Type: {garment_type}
Body Shape: {body_shape}
Available Sizes: {', '.join(available_sizes)}

Use your expert knowledge of international sizing standards.
Do NOT apply fixed thresholds — reason holistically from ALL measurements (chest, waist, hips, shoulder width, height, etc.).
Consider the garment type and body shape when deciding between borderline sizes.

Return ONLY a valid JSON object:
{{
    "recommended_size": "<one of the available sizes>",
    "fit_type": "<slim, regular, or oversize>",
    "reasoning": "<one sentence explaining why this size fits best>"
}}"""

        response = self.model.generate_content(prompt)
        result = self._parse_json_response(response.text)

        if not result or "recommended_size" not in result:
            raise ValueError("Gemini failed to return a valid size recommendation")

        recommended = result["recommended_size"]
        if recommended not in available_sizes:
            raise ValueError(
                f"Gemini returned size '{recommended}' which is not in available sizes {available_sizes}"
            )

        if result.get("fit_type") not in ["slim", "regular", "oversize"]:
            result["fit_type"] = "regular"

        logger.info(f"Gemini size recommendation: {recommended} ({result.get('reasoning', '')})")
        return result

    def get_color_recommendations(self, skin_tone: str, undertone: str = "warm", body_shape: str = "rectangle", garment_type: str = None) -> Dict:
        """
        Ask Gemini to pick exactly ONE shirt colour and ONE pants colour
        from the pre-approved palette for the given skin tone.

        Returns:
            dict with keys 'recommended_shirt' and 'recommended_pants'
            (both are exact colour name strings from color_palettes.py).
        """
        if not self.available:
            raise RuntimeError("Gemini AI is not available for color recommendations")

        from fitting_system.color_palettes import get_shirt_color_names, get_pants_color_names

        shirt_options = get_shirt_color_names(skin_tone)
        pants_options = get_pants_color_names(skin_tone)

        prompt = f"""You are an expert fashion color consultant.

Person's profile:
- Skin tone: {skin_tone}
- Undertone: {undertone}
- Body shape: {body_shape}

Available shirt colors (pick exactly ONE): {', '.join(shirt_options)}
Available pants colors (pick exactly ONE): {', '.join(pants_options)}

Choose the single best shirt color and single best pants color from the lists above.
You MUST only use color names from the provided lists. Do NOT invent new colors.

Return ONLY a JSON object:
{{"recommended_shirt": "<one of the shirt colors>", "recommended_pants": "<one of the pants colors>"}}"""

        response = self.model.generate_content(prompt)
        result = self._parse_json_response(response.text)

        rec_shirt = result.get("recommended_shirt", "")
        rec_pants = result.get("recommended_pants", "")

        # Validate – fall back to first option if Gemini hallucinated
        if rec_shirt not in shirt_options:
            logger.warning(f"Gemini returned invalid shirt color '{rec_shirt}', falling back to '{shirt_options[0]}'")
            rec_shirt = shirt_options[0]
        if rec_pants not in pants_options:
            logger.warning(f"Gemini returned invalid pants color '{rec_pants}', falling back to '{pants_options[0]}'")
            rec_pants = pants_options[0]

        return {"recommended_shirt": rec_shirt, "recommended_pants": rec_pants}

    def get_styling_advice(self, measurements: Dict, body_shape: str, skin_tone: str, undertone: str = "warm") -> str:
        if not self.available:
            raise RuntimeError("Gemini AI is not available for styling advice")

        prompt = f"""You are a personal fashion stylist. Give brief, practical styling advice.

Person's profile:
- Body shape: {body_shape}
- Skin tone: {skin_tone} with {undertone} undertone
- Measurements: {json.dumps(measurements)}

Give 3-4 concise styling tips specific to their body type and coloring. Return plain text."""

        response = self.model.generate_content(prompt)
        return response.text.strip()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_gemini_client_instance = None


def get_gemini_client() -> GeminiClient:
    """Return the global GeminiClient singleton."""
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = GeminiClient()
    return _gemini_client_instance