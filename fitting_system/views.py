from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.utils.translation import gettext as _
import json
import base64
import re
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


SKIN_TONE_LABELS = {
    'very_light': _("Very Light"),
    'light': _("Light"),
    'intermediate': _("Intermediate"),
    'tan': _("Tan"),
    'dark': _("Dark"),
}

UNDERTONE_LABELS = {
    'warm': _("Warm"),
    'cool': _("Cool"),
}


def _translate_dynamic_label(value: str) -> str:
    """
    Translate dynamic labels from DB.
    Handles both plain strings and mistakenly stored template tags
    like "{% trans 'AVAILABLE COLOR' %}".
    """
    if not value:
        return value
    cleaned = value.strip()
    m = re.fullmatch(r"\{\%\s*trans\s+['\"](.+?)['\"]\s*\%\}", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    # Be tolerant of malformed template-like strings saved in DB
    elif '{%' in cleaned and '%}' in cleaned:
        q = re.search(r"['\"](.+?)['\"]", cleaned)
        if q:
            cleaned = q.group(1).strip()
    return _(cleaned)


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
        return JsonResponse({'error': _('POST method required')}, status=400)

    try:
        data       = json.loads(request.body)
        image_data = data.get('image')
        mode       = data.get('mode', 'body')

        if not image_data:
            return JsonResponse({'error': _('Image is required')}, status=400)

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
        return JsonResponse({'error': _('POST method required')}, status=400)

    try:
        data = json.loads(request.body)

        front_image_data = data.get('front_image')
        face_image_data  = data.get('face_image')
        side_image_data  = data.get('side_image')   # kept for backward compat
        user_height_cm   = data.get('user_height_cm')

        if not front_image_data:
            return JsonResponse({'error': _('Front (body) image is required')}, status=400)
        if not face_image_data:
            return JsonResponse({'error': _('Face image is required for skin tone detection')}, status=400)

        # Validate user height
        if user_height_cm is None:
            return JsonResponse({'error': _('Height is required. Please enter your height in cm.')}, status=400)
        try:
            user_height_cm = float(user_height_cm)
        except (TypeError, ValueError):
            return JsonResponse({'error': _('Height must be a valid number in cm.')}, status=400)
        if user_height_cm < 100 or user_height_cm > 250:
            return JsonResponse({'error': _('Height must be between 100 and 250 cm.')}, status=400)

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
            'skin_tone_display': SKIN_TONE_LABELS.get(skin_tone, skin_tone.replace('_', ' ').title()),
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
        return JsonResponse({'error': _('Processing failed: %(error)s') % {'error': str(e)}}, status=500)


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
        return JsonResponse({'error': _('POST method required')}, status=400)

    try:
        data = json.loads(request.body)

        hand_image_data = data.get('hand_image')
        measurements    = data.get('measurements', {})

        if not hand_image_data:
            return JsonResponse({'error': _('Hand image is required for skin tone detection')}, status=400)

        # Validate required measurements
        required_fields = ['height', 'chest', 'waist', 'hip']
        field_labels = {
            'height': _("Height"),
            'chest': _("Bust / Chest"),
            'waist': _("Waist"),
            'hip': _("Hip"),
        }
        for field in required_fields:
            val = measurements.get(field)
            if val is None:
                return JsonResponse({'error': _('%(field)s is required.') % {'field': field_labels.get(field, field.title())}}, status=400)
            try:
                measurements[field] = float(val)
            except (TypeError, ValueError):
                return JsonResponse({'error': _('%(field)s must be a valid number.') % {'field': field_labels.get(field, field.title())}}, status=400)

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
            return JsonResponse({'error': _('Height must be between 100 and 250 cm.')}, status=400)

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
            'skin_tone_display': SKIN_TONE_LABELS.get(skin_tone, skin_tone.replace('_', ' ').title()),
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
        return JsonResponse({'error': _('Processing failed: %(error)s') % {'error': str(e)}}, status=500)


# ---------------------------------------------------------------------------
# Recommendations page
# ---------------------------------------------------------------------------

def recommendations(request, session_id):
    """Display AI results and matching store products."""
    body_scan = get_object_or_404(BodyScan, session_id=session_id)

    # Retrieve the LLM-recommended size from the first Recommendation record
    first_rec = body_scan.recommendations.first()
    recommended_size = first_rec.recommended_size if first_rec else 'M'

    # Enforce scan-derived gender so recommendations are gender-specific.
    # frame_count == 0 is used by this project for women/manual flow.
    gender = 'women' if body_scan.frame_count == 0 else 'men'

    # Get skin-tone-recommended color names so product cards sync with avatar
    from .color_palettes import get_shirt_color_names, get_pants_color_names
    preferred_colors = (
        get_shirt_color_names(body_scan.skin_tone)
        + get_pants_color_names(body_scan.skin_tone)
    )

    # Match products whose variants have the recommended size in stock
    matching_products = _get_matching_products(
        recommended_size, gender, preferred_colors=preferred_colors, limit=12,
    )

    context = {
        'body_scan':         body_scan,
        'recommended_size':  recommended_size,
        'skin_tone_display': SKIN_TONE_LABELS.get(
            body_scan.skin_tone,
            body_scan.skin_tone.replace('_', ' ').title(),
        ),
        'undertone_display': UNDERTONE_LABELS.get(
            body_scan.undertone,
            body_scan.undertone.title(),
        ),
        'matching_products': matching_products,
        'selected_gender':   gender,
    }
    return render(request, 'recommendations.html', context)


def _get_matching_products(recommended_size: str, gender=None, preferred_colors=None, limit=12):
    """
    Return products that have the recommended size in stock.
    Each item in the list is a dict with product + variant info.
    When preferred_colors is provided, prefer variants whose color matches
    the skin-tone palette so product cards sync with the avatar.
    """
    category_labels = {
        'shirt': _("Shirt"),
        'pants': _("Pants"),
        'jacket': _("Jacket"),
        'dress': _("Dress"),
        'skirt': _("Skirt"),
        't-shirt': _("T-Shirt"),
        'tshirt': _("T-Shirt"),
        'jeans': _("Jeans"),
    }

    if gender and gender in ['men', 'women']:
        products = Product.objects.filter(gender=gender)
    else:
        products = Product.objects.all()

    preferred_set = set(preferred_colors) if preferred_colors else set()

    results = []
    for product in products:
        base_qs = ProductVariant.objects.filter(
            product=product,
            size__name=recommended_size,
            inventory__quantity__gt=0,
        ).select_related('size', 'color', 'product')

        # Try to find a variant whose color is in the preferred palette first
        variant = None
        if preferred_set:
            variant = base_qs.filter(color__name__in=preferred_set).first()
        # Fall back to any available variant
        if not variant:
            variant = base_qs.first()

        if variant:
            results.append({
                'product':          product,
                'product_name':     _(product.name),
                'category_label':   category_labels.get(product.category, product.category.title()),
                'variant':          variant,
                'recommended_size': recommended_size,
                'color_name':       _translate_dynamic_label(variant.color.name),
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

    # Determine gender
    gender = request.GET.get('gender', None)
    if not gender:
        gender = 'women' if body_scan.frame_count == 0 else 'men'

    # Map skin tone to avatar swatch
    skin_info = SKIN_TONE_MAP.get(body_scan.skin_tone, SKIN_TONE_MAP['intermediate'])

    # ── Unified colour palette for this skin tone ──
    from .color_palettes import get_shirt_colors, get_pants_colors, SKIN_TONE_PALETTES
    import json as _json

    skin_tone_key = body_scan.skin_tone
    shirt_palette = get_shirt_colors(skin_tone_key)
    pants_palette = get_pants_colors(skin_tone_key)

    # Try to get Gemini-recommended defaults
    rec_shirt_hex = shirt_palette[0]['hex']  # fallback to first
    rec_pants_hex = pants_palette[0]['hex']  # fallback to first

    try:
        from .ai_modules.gemini_client import get_gemini_client
        gemini = get_gemini_client()
        if gemini.available:
            color_rec = gemini.get_color_recommendations(
                skin_tone=skin_tone_key,
                undertone=body_scan.undertone,
            )
            rec_shirt_name = color_rec.get('recommended_shirt', '')
            rec_pants_name = color_rec.get('recommended_pants', '')
            # Map names → hex codes
            for c in shirt_palette:
                if c['name'] == rec_shirt_name:
                    rec_shirt_hex = c['hex']
                    break
            for c in pants_palette:
                if c['name'] == rec_pants_name:
                    rec_pants_hex = c['hex']
                    break
    except Exception as e:
        logger.warning(f"Failed to get Gemini color recommendation for avatar: {e}")

    # Build full palettes JSON for all 5 skin tones (for skin-swatch switching)
    palettes_for_js = {}
    for st_key, palette in SKIN_TONE_PALETTES.items():
        palettes_for_js[st_key] = {
            'shirts': [{'hex': c['hex'].lstrip('#'), 'tip': c['name']} for c in palette['shirts']],
            'pants':  [{'hex': c['hex'].lstrip('#'), 'tip': c['name']} for c in palette['pants']],
        }

    context = {
        'body_scan':        body_scan,
        'session_id':       str(session_id),
        'recommended_size': recommended_size,
        'gender':           gender,
        'skin_tone_index':  skin_info['index'],
        'skin_tone_hex':    skin_info['hex'],
        'skin_tone_label':  skin_info['label'],
        'skin_tone_display': SKIN_TONE_LABELS.get(
            body_scan.skin_tone,
            body_scan.skin_tone.replace('_', ' ').title(),
        ),
        'undertone_display': UNDERTONE_LABELS.get(
            body_scan.undertone,
            body_scan.undertone.title(),
        ),
        'skin_tone_key':    skin_tone_key,
        'rec_shirt_hex':    rec_shirt_hex.lstrip('#'),
        'rec_pants_hex':    rec_pants_hex.lstrip('#'),
        'palettes_json':    _json.dumps(palettes_for_js),
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

    category_labels = {
        'shirt': _("Shirt"),
        'pants': _("Pants"),
        'jacket': _("Jacket"),
        'dress': _("Dress"),
        'skirt': _("Skirt"),
        't-shirt': _("T-Shirt"),
        'tshirt': _("T-Shirt"),
        'jeans': _("Jeans"),
    }
    gender_labels = {
        'men': _("Men"),
        'women': _("Women"),
        'unisex': _("Unisex"),
    }

    products = list(products)
    for product in products:
        product.localized_name = _(product.name)
        product.localized_description = _(product.description)
        product.localized_category = category_labels.get(product.category, product.category.title())
        product.localized_gender = gender_labels.get(product.gender, product.gender.title())

    categories = Product.objects.values_list('category', flat=True).distinct().order_by('category')
    genders    = Product.objects.values_list('gender', flat=True).distinct().order_by('gender')

    category_options = [
        {'value': c, 'label': category_labels.get(c, c.title())}
        for c in categories
    ]
    gender_options = [
        {'value': g, 'label': gender_labels.get(g, g.title())}
        for g in genders
    ]

    context = {
        'products':         products,
        'categories':       categories,
        'genders':          genders,
        'category_options': category_options,
        'gender_options':   gender_options,
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

    for variant in variants:
        try:
            if variant.inventory.is_available:
                available_sizes.add(variant.size)
        except Inventory.DoesNotExist:
            pass

    related_products = Product.objects.filter(
        category=product.category
    ).exclude(id=product.id)[:4]

    context = {
        'product':          product,
        'variants':         variants,
        'available_sizes':  sorted(available_sizes, key=lambda x: x.id),
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
