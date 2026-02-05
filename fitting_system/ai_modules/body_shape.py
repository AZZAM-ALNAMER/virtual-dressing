"""
Body Shape Classification Module
Classifies body shape based on measurement ratios for fashion-grade recommendations
"""
from typing import Dict


# Body shape archetypes with descriptions
BODY_SHAPES = {
    'hourglass': 'Balanced bust/hip with defined waist',
    'rectangle': 'Similar bust/waist/hip measurements',
    'triangle': 'Hips wider than shoulders (pear shape)',
    'inverted_triangle': 'Shoulders wider than hips (athletic)',
    'oval': 'Fuller midsection (apple shape)',
}


def classify_body_shape(measurements: Dict[str, float]) -> str:
    """
    Classify body shape based on bust-waist-hip and shoulder-hip ratios.
    Fashion brands use this to adjust fit recommendations.
    
    Args:
        measurements: Dict with 'chest', 'waist', 'hip', 'shoulder_width'
        
    Returns:
        Body shape string: 'hourglass', 'rectangle', 'triangle', 
                          'inverted_triangle', or 'oval'
    """
    bust = measurements.get('chest', 0)
    waist = measurements.get('waist', 0)
    hip = measurements.get('hip', 0)
    shoulder = measurements.get('shoulder_width', 0)
    
    # Fallback if missing critical measurements
    if waist == 0 or hip == 0:
        return 'rectangle'
    
    # Calculate key ratios
    waist_to_hip = waist / hip
    waist_to_bust = waist / bust if bust > 0 else 1
    shoulder_to_hip = shoulder / hip if hip > 0 else 1
    
    # Classification logic based on fashion industry standards
    # Hourglass: bust and hip similar, waist significantly smaller
    if waist_to_hip <= 0.75 and waist_to_bust <= 0.75:
        return 'hourglass'
    
    # Inverted Triangle: shoulders notably wider than hips
    elif shoulder_to_hip > 1.05:
        return 'inverted_triangle'
    
    # Triangle (Pear): hips notably wider than shoulders
    elif shoulder_to_hip < 0.95:
        return 'triangle'
    
    # Oval (Apple): waist approaches or exceeds hip measurement
    elif waist_to_hip > 0.85:
        return 'oval'
    
    # Rectangle: relatively straight up and down
    else:
        return 'rectangle'


def get_body_shape_display(body_shape: str) -> str:
    """Get human-readable display name for body shape."""
    display_names = {
        'hourglass': 'Hourglass',
        'rectangle': 'Rectangle',
        'triangle': 'Triangle (Pear)',
        'inverted_triangle': 'Inverted Triangle (Athletic)',
        'oval': 'Oval (Apple)',
    }
    return display_names.get(body_shape, body_shape.replace('_', ' ').title())


def get_body_shape_description(body_shape: str) -> str:
    """Get description for body shape."""
    return BODY_SHAPES.get(body_shape, '')
