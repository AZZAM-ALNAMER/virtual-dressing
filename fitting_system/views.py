from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
import json
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

from .models import Product, ProductVariant, Inventory, BodyScan, Recommendation, Size, Color
from .ai_modules.yolo_analyzer import get_yolo_analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_base64_image(base64_string):
    """Decode base64 image to numpy array (BGR)."""
    if not base64_string:
        return None
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    pil_image  = Image.open(BytesIO(image_data))
    image_array = np.array(pil_image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------

def index(request):
    """Landing page"""
    return render(request, 'index.html')


def scan(request):
    """Camera interface and scanning workflow"""
    return render(request, 'scan.html')


# ---------------------------------------------------------------------------
# API: real-time frame analysis
# ---------------------------------------------------------------------------

@csrf_exempt
def analyze_frame(request):
    """
    Analyse a single camera frame for real-time feedback.
    mode='body'  → YOLO pose detection (body step)
    mode='face'  → face detection (skin-tone selfie step)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)

    try:
        data       = json.loads(request.body)
        image_data = data.get('image')
        mode       = data.get('mode', 'body')

        if not image_data:
            return JsonResponse({'error': 'Image is required'}, status=400)

        image   = decode_base64_image(image_data)
        analyzer = get_yolo_analyzer()

        if mode == 'face':
            result = analyzer.analyze_face_frame(image)
        else:
            result = analyzer.analyze_pose_frame(image)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e), 'detected': False}, status=500)


# ---------------------------------------------------------------------------
# API: process full scan (body image + face image)
# ---------------------------------------------------------------------------

@csrf_exempt
def process_scan(request):
    """
    Process two captured images:
        front_image  – full-body front view  → YOLO pose + measurements
        face_image   – close-up selfie       → skin-tone extraction
    Then ask the LLM for a recommended size letter.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)

    try:
        data = json.loads(request.body)

        front_image_data = data.get('front_image')
        face_image_data  = data.get('face_image')
        side_image_data  = data.get('side_image')   # kept for backward compat
        user_height_cm   = data.get('user_height_cm')

        if not front_image_data:
            return JsonResponse({'error': 'Front (body) image is required'}, status=400)
        if not face_image_data:
            return JsonResponse({'error': 'Face image is required for skin tone detection'}, status=400)

        # Validate user height
        if user_height_cm is None:
            return JsonResponse({'error': 'Height is required. Please enter your height in cm.'}, status=400)
        try:
            user_height_cm = float(user_height_cm)
        except (TypeError, ValueError):
            return JsonResponse({'error': 'Height must be a valid number in cm.'}, status=400)
        if user_height_cm < 100 or user_height_cm > 250:
            return JsonResponse({'error': 'Height must be between 100 and 250 cm.'}, status=400)

        front_image = decode_base64_image(front_image_data)
        face_image  = decode_base64_image(face_image_data)

        analyzer = get_yolo_analyzer()

        # Full pipeline: measurements + skin tone + LLM size
        analysis = analyzer.full_analysis(
            body_image_bgr=front_image,
            face_image_bgr=face_image,
            user_height_cm=user_height_cm,
        )

        measurements     = analysis['measurements']
        skin_tone        = analysis['skin_tone']
        undertone        = analysis['undertone']
        recommended_size = analysis['recommended_size']
        confidence       = analysis.get('confidence', 0.85)

        # Persist to DB
        body_scan = BodyScan.objects.create(
            height          = measurements.get('height', 170),
            shoulder_width  = measurements.get('shoulder_width', 42),
            chest           = measurements.get('chest', 92),
            waist           = measurements.get('waist', 78),
            hip             = measurements.get('hip'),
            torso_length    = measurements.get('torso_length'),
            arm_length      = measurements.get('arm_length'),
            inseam          = measurements.get('inseam'),
            body_shape      = 'rectangle',   # not used in new flow
            skin_tone       = skin_tone,
            undertone       = undertone,
            confidence_score= confidence,
            frame_count     = 1,
        )

        # Store the LLM-recommended size as a Recommendation record
        # (we create one generic record so the recommendations view can read it)
        try:
            Recommendation.objects.create(
                body_scan         = body_scan,
                product           = Product.objects.first(),   # placeholder
                recommended_size  = recommended_size,
                recommended_fit   = 'regular',
                recommended_colors= '',
                priority          = 100,
            )
        except Exception:
            pass   # no products in DB yet – that's fine

        return JsonResponse({
            'success':          True,
            'session_id':       str(body_scan.session_id),
            'skin_tone':        skin_tone,
            'skin_tone_display': skin_tone.replace('_', ' ').title(),
            'undertone':        undertone,
            'recommended_size': recommended_size,
            'confidence':       round(confidence, 2),
        })

    except ValueError as e:
        logger.error(f"Validation error in process_scan: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    except RuntimeError as e:
        logger.error(f"Runtime error in process_scan: {e}")
        return JsonResponse({'error': str(e)}, status=503)
    except Exception as e:
        logger.exception(f"Unexpected error in process_scan: {e}")
        return JsonResponse({'error': f'Processing failed: {str(e)}'}, status=500)


# ---------------------------------------------------------------------------
# API: process women scan (manual measurements + hand image)
# ---------------------------------------------------------------------------

@csrf_exempt
def process_scan_women(request):
    """
    Process women's scan:
        hand_image       – photo of hand → skin-tone extraction
        measurements     – manually entered body measurements (dict)
    Then ask the LLM for a recommended size letter.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)

    try:
        data = json.loads(request.body)

        hand_image_data = data.get('hand_image')
        measurements    = data.get('measurements', {})

        if not hand_image_data:
            return JsonResponse({'error': 'Hand image is required for skin tone detection'}, status=400)

        # Validate required measurements
        required_fields = ['height', 'chest', 'waist', 'hip']
        for field in required_fields:
            val = measurements.get(field)
            if val is None:
                return JsonResponse({'error': f'{field.title()} is required.'}, status=400)
            try:
                measurements[field] = float(val)
            except (TypeError, ValueError):
                return JsonResponse({'error': f'{field.title()} must be a valid number.'}, status=400)

        # Convert optional fields to float if present
        optional_fields = ['shoulder_width', 'inseam', 'arm_length', 'torso_length']
        for field in optional_fields:
            val = measurements.get(field)
            if val is not None and val != '':
                try:
                    measurements[field] = float(val)
                except (TypeError, ValueError):
                    measurements.pop(field, None)
            else:
                measurements.pop(field, None)

        # Validate height range
        height = measurements['height']
        if height < 100 or height > 250:
            return JsonResponse({'error': 'Height must be between 100 and 250 cm.'}, status=400)

        hand_image = decode_base64_image(hand_image_data)

        analyzer = get_yolo_analyzer()

        # Women pipeline: manual measurements + hand skin tone + LLM size
        analysis = analyzer.women_analysis(
            measurements=measurements,
            hand_image_bgr=hand_image,
        )

        skin_tone        = analysis['skin_tone']
        undertone        = analysis['undertone']
        recommended_size = analysis['recommended_size']
        confidence       = analysis.get('confidence', 0.90)

        # Persist to DB
        body_scan = BodyScan.objects.create(
            height          = measurements.get('height', 170),
            shoulder_width  = measurements.get('shoulder_width', 0),
            chest           = measurements.get('chest', 0),
            waist           = measurements.get('waist', 0),
            hip             = measurements.get('hip'),
            torso_length    = measurements.get('torso_length'),
            arm_length      = measurements.get('arm_length'),
            inseam          = measurements.get('inseam'),
            body_shape      = 'hourglass',
            skin_tone       = skin_tone,
            undertone       = undertone,
            confidence_score= confidence,
            frame_count     = 0,
        )

        # Store as Recommendation record
        try:
            Recommendation.objects.create(
                body_scan         = body_scan,
                product           = Product.objects.first(),
                recommended_size  = recommended_size,
                recommended_fit   = 'regular',
                recommended_colors= '',
                priority          = 100,
            )
        except Exception:
            pass

        return JsonResponse({
            'success':          True,
            'session_id':       str(body_scan.session_id),
            'skin_tone':        skin_tone,
            'skin_tone_display': skin_tone.replace('_', ' ').title(),
            'undertone':        undertone,
            'recommended_size': recommended_size,
            'confidence':       round(confidence, 2),
        })

    except ValueError as e:
        logger.error(f"Validation error in process_scan_women: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    except RuntimeError as e:
        logger.error(f"Runtime error in process_scan_women: {e}")
        return JsonResponse({'error': str(e)}, status=503)
    except Exception as e:
        logger.exception(f"Unexpected error in process_scan_women: {e}")
        return JsonResponse({'error': f'Processing failed: {str(e)}'}, status=500)


# ---------------------------------------------------------------------------
# Recommendations page
# ---------------------------------------------------------------------------

def recommendations(request, session_id):
    """Display AI results and matching store products."""
    body_scan = get_object_or_404(BodyScan, session_id=session_id)

    # Retrieve the LLM-recommended size from the first Recommendation record
    first_rec = body_scan.recommendations.first()
    recommended_size = first_rec.recommended_size if first_rec else 'M'

    # Gender filter
    gender = request.GET.get('gender', None)
    if gender and gender not in ['men', 'women']:
        gender = None

    # Match products whose variants have the recommended size in stock
    matching_products = _get_matching_products(recommended_size, gender, limit=12)

    context = {
        'body_scan':         body_scan,
        'recommended_size':  recommended_size,
        'skin_tone_display': body_scan.skin_tone.replace('_', ' ').title(),
        'undertone_display': body_scan.undertone.title(),
        'matching_products': matching_products,
        'selected_gender':   gender,
    }
    return render(request, 'recommendations.html', context)


def _get_matching_products(recommended_size: str, gender=None, limit=12):
    """
    Return products that have the recommended size in stock.
    Each item in the list is a dict with product + variant info.
    """
    if gender and gender in ['men', 'women']:
        products = Product.objects.filter(Q(gender=gender) | Q(gender='unisex'))
    else:
        products = Product.objects.all()

    results = []
    for product in products:
        variant = ProductVariant.objects.filter(
            product=product,
            size__name=recommended_size,
            inventory__quantity__gt=0,
        ).select_related('size', 'color', 'product').first()

        if variant:
            results.append({
                'product':          product,
                'variant':          variant,
                'recommended_size': recommended_size,
                'color_name':       variant.color.name,
                'color_hex':        variant.color.hex_code,
            })

    return results[:limit]


# ---------------------------------------------------------------------------
# Avatar page (3D avatar viewer with skin tone & color recommendations)
# ---------------------------------------------------------------------------

# Map BodyScan skin_tone values → avatar swatch index & hex
SKIN_TONE_MAP = {
    'very_light':   {'index': 0, 'hex': 'fde8d0', 'label': 'Porcelain'},
    'light':        {'index': 1, 'hex': 'f5cba7', 'label': 'Ivory'},
    'intermediate': {'index': 2, 'hex': 'e8a87c', 'label': 'Peach'},
    'tan':          {'index': 3, 'hex': 'c68642', 'label': 'Tan'},
    'dark':         {'index': 4, 'hex': '8d5524', 'label': 'Brown'},
}


def avatar(request, session_id):
    """3D avatar page with skin tone and color recommendations."""
    body_scan = get_object_or_404(BodyScan, session_id=session_id)

    # Retrieve recommended size
    first_rec = body_scan.recommendations.first()
    recommended_size = first_rec.recommended_size if first_rec else 'M'

    # Determine gender: women scans use frame_count=0, men use frame_count>=1
    # Also allow override via query param
    gender = request.GET.get('gender', None)
    if not gender:
        gender = 'women' if body_scan.frame_count == 0 else 'men'

    # Map skin tone to avatar swatch
    skin_info = SKIN_TONE_MAP.get(body_scan.skin_tone, SKIN_TONE_MAP['intermediate'])

    context = {
        'body_scan':        body_scan,
        'session_id':       str(session_id),
        'recommended_size': recommended_size,
        'gender':           gender,
        'skin_tone_index':  skin_info['index'],
        'skin_tone_hex':    skin_info['hex'],
        'skin_tone_label':  skin_info['label'],
        'skin_tone_display': body_scan.skin_tone.replace('_', ' ').title(),
        'undertone_display': body_scan.undertone.title(),
    }
    return render(request, 'avatar.html', context)


# ---------------------------------------------------------------------------
# Store & inventory views (unchanged)
# ---------------------------------------------------------------------------

def inventory_dashboard(request):
    """Inventory management dashboard"""
    variants = ProductVariant.objects.select_related(
        'product', 'size', 'color', 'inventory'
    ).all()

    in_stock, low_stock, out_of_stock = [], [], []

    for variant in variants:
        try:
            if variant.inventory.is_out_of_stock:
                out_of_stock.append(variant)
            elif variant.inventory.is_low_stock:
                low_stock.append(variant)
            else:
                in_stock.append(variant)
        except Inventory.DoesNotExist:
            out_of_stock.append(variant)

    context = {
        'in_stock':       in_stock,
        'low_stock':      low_stock,
        'out_of_stock':   out_of_stock,
        'total_variants': variants.count(),
    }
    return render(request, 'inventory.html', context)


def store(request):
    """Online store product catalog"""
    category = request.GET.get('category', '')
    gender   = request.GET.get('gender', '')
    search   = request.GET.get('search', '')

    products = Product.objects.all()

    if category:
        products = products.filter(category=category)
    if gender:
        products = products.filter(Q(gender=gender) | Q(gender='unisex'))
    if search:
        products = products.filter(
            Q(name__icontains=search) | Q(description__icontains=search)
        )

    categories = Product.objects.values_list('category', flat=True).distinct().order_by('category')
    genders    = Product.objects.values_list('gender', flat=True).distinct().order_by('gender')

    context = {
        'products':         products,
        'categories':       categories,
        'genders':          genders,
        'selected_category': category,
        'selected_gender':  gender,
        'search_query':     search,
    }
    return render(request, 'store.html', context)


def product_detail(request, product_id):
    """Product detail page"""
    product  = get_object_or_404(Product, id=product_id)
    variants = product.variants.select_related('size', 'color', 'inventory').all()

    available_sizes  = set()
    available_colors = set()

    for variant in variants:
        try:
            if variant.inventory.is_available:
                available_sizes.add(variant.size)
                available_colors.add(variant.color)
        except Inventory.DoesNotExist:
            pass

    related_products = Product.objects.filter(
        category=product.category
    ).exclude(id=product.id)[:4]

    context = {
        'product':          product,
        'variants':         variants,
        'available_sizes':  sorted(available_sizes, key=lambda x: x.id),
        'available_colors': sorted(available_colors, key=lambda x: x.name),
        'related_products': related_products,
    }
    return render(request, 'product_detail.html', context)


def api_inventory(request):
    """API endpoint for inventory data"""
    variants = ProductVariant.objects.select_related(
        'product', 'size', 'color', 'inventory'
    ).all()

    data = []
    for variant in variants:
        try:
            inv = variant.inventory
            data.append({
                'id':            variant.id,
                'product':       variant.product.name,
                'size':          variant.size.name,
                'color':         variant.color.name,
                'quantity':      inv.quantity,
                'is_low_stock':  inv.is_low_stock,
                'is_out_of_stock': inv.is_out_of_stock,
            })
        except Inventory.DoesNotExist:
            data.append({
                'id':            variant.id,
                'product':       variant.product.name,
                'size':          variant.size.name,
                'color':         variant.color.name,
                'quantity':      0,
                'is_low_stock':  False,
                'is_out_of_stock': True,
            })

    return JsonResponse({'inventory': data})
