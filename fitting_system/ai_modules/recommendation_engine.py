"""
Recommendation Engine
Generates intelligent clothing recommendations using Gemini AI
and database-backed product matching.

Architecture:
    - Gemini AI: Intelligent size/fit/color recommendations
    - Database: Product matching, inventory checking, variant selection
    - color_palettes.py: Single source of truth for colour options
"""

import logging
from typing import Dict, List, Tuple
from django.db.models import Q

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generates clothing recommendations using Gemini AI for intelligence
    and the product database for matching.

    Gemini handles:
        - Size recommendation (considering body shape + garment type)
        - Fit type recommendation
        - Color recommendations (1 shirt + 1 pants from the unified palette)
        - Styling advice

    Database handles:
        - Finding products with matching size in stock
        - Color variant matching
        - Inventory availability
    """

    SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']

    GARMENT_MEASUREMENTS = {
        'shirt': {'fit_focus': 'chest'},
        'pants': {'fit_focus': 'waist'},
        'dress': {'fit_focus': 'waist'},
        'jacket': {'fit_focus': 'chest'},
        'skirt': {'fit_focus': 'waist'},
    }

    def __init__(self):
        self._gemini = None

    @property
    def gemini(self):
        """Lazy-load Gemini client."""
        if self._gemini is None:
            from .gemini_client import get_gemini_client
            self._gemini = get_gemini_client()
        return self._gemini

    # ── Size ──────────────────────────────────────────────────────
    def recommend_size(self, measurements: Dict[str, float],
                       garment_type: str = 'shirt',
                       body_shape: str = 'rectangle') -> str:
        if not self.gemini.available:
            raise RuntimeError("Gemini AI is not available for size recommendation")
        result = self.gemini.get_size_recommendation(
            measurements=measurements,
            garment_type=garment_type,
            body_shape=body_shape,
        )
        size = result.get("recommended_size", "M")
        logger.info(f"Gemini size recommendation: {size} for {garment_type}")
        return size

    # ── Fit ───────────────────────────────────────────────────────
    def recommend_fit(self, measurements: Dict[str, float],
                      garment_type: str = 'shirt',
                      body_shape: str = 'rectangle') -> str:
        if not self.gemini.available:
            raise RuntimeError("Gemini AI is not available for fit recommendation")
        result = self.gemini.get_size_recommendation(
            measurements=measurements,
            garment_type=garment_type,
            body_shape=body_shape,
        )
        fit = result.get("fit_type", "regular")
        return fit if fit in ["slim", "regular", "oversize"] else "regular"

    # ── Colours (NEW: returns dict with shirt + pants) ────────────
    def recommend_colors(self, skin_tone: str, undertone: str = 'warm') -> Dict[str, str]:
        """
        Returns dict: {'recommended_shirt': '<name>', 'recommended_pants': '<name>'}
        """
        if not self.gemini.available:
            raise RuntimeError("Gemini AI is not available for color recommendation")
        result = self.gemini.get_color_recommendations(
            skin_tone=skin_tone,
            undertone=undertone,
        )
        if not result or 'recommended_shirt' not in result:
            raise ValueError("Gemini returned empty color recommendations")
        return result

    # ── Full product matching ─────────────────────────────────────
    def get_matching_product_variants(
        self,
        body_scan,
        gender: str = None,
        limit: int = 6
    ) -> List[Dict]:
        """
        Get actual products from store with specific size and color recommendations.
        Uses Gemini AI for size/color/fit recommendations, then matches against inventory.
        """
        from fitting_system.models import Product, ProductVariant, Color, Size

        measurements = {
            'height': float(body_scan.height),
            'chest': float(body_scan.chest),
            'waist': float(body_scan.waist),
            'shoulder_width': float(body_scan.shoulder_width),
        }
        if body_scan.hip:
            measurements['hip'] = float(body_scan.hip)
        if body_scan.inseam:
            measurements['inseam'] = float(body_scan.inseam)
        if body_scan.torso_length:
            measurements['torso_length'] = float(body_scan.torso_length)
        if body_scan.arm_length:
            measurements['arm_length'] = float(body_scan.arm_length)

        body_shape = getattr(body_scan, 'body_shape', 'rectangle') or 'rectangle'
        undertone = getattr(body_scan, 'undertone', 'warm')

        # Gemini-powered recommendations
        color_rec = self.recommend_colors(body_scan.skin_tone, undertone)
        rec_shirt_name = color_rec['recommended_shirt']
        rec_pants_name = color_rec['recommended_pants']
        recommended_fit = self.recommend_fit(measurements, body_shape=body_shape)

        # Map colour names → Color objects
        rec_shirt_color = Color.objects.filter(name=rec_shirt_name).first()
        rec_pants_color = Color.objects.filter(name=rec_pants_name).first()

        # Filter products
        if gender and gender in ['men', 'women']:
            products = Product.objects.filter(
                Q(gender=gender) | Q(gender='unisex')
            )
        else:
            products = Product.objects.all()

        matching_products = []

        for product in products:
            rec_size = self.recommend_size(
                measurements,
                garment_type=product.category,
                body_shape=body_shape,
            )

            # Choose the right recommended colour for the product category
            is_top = product.category in ('shirt', 'jacket', 'dress')
            target_color = rec_shirt_color if is_top else rec_pants_color
            target_color_name = rec_shirt_name if is_top else rec_pants_name

            fit_matches = product.fit_type == recommended_fit

            # Priority 1: Exact size + recommended colour + in stock
            if target_color:
                variant = ProductVariant.objects.filter(
                    product=product,
                    size__name=rec_size,
                    color=target_color,
                    inventory__quantity__gt=0,
                ).select_related('size', 'color', 'product').first()

                if variant:
                    matching_products.append({
                        'product': product,
                        'variant': variant,
                        'recommended_size': rec_size,
                        'recommended_color': variant.color.name,
                        'color_hex': variant.color.hex_code,
                        'fit_type': product.fit_type,
                        'is_perfect_match': True,
                        'fit_matches_recommendation': fit_matches,
                        'recommended_fit': recommended_fit,
                        'fit_message': f"This {product.category} in size {rec_size} with {variant.color.name} will fit you perfectly!",
                    })
                    continue

            # Priority 2: Exact size + any colour in stock
            fallback_variant = ProductVariant.objects.filter(
                product=product,
                size__name=rec_size,
                inventory__quantity__gt=0,
            ).select_related('size', 'color', 'product').first()

            if fallback_variant:
                matching_products.append({
                    'product': product,
                    'variant': fallback_variant,
                    'recommended_size': rec_size,
                    'recommended_color': fallback_variant.color.name,
                    'color_hex': fallback_variant.color.hex_code,
                    'fit_type': product.fit_type,
                    'is_perfect_match': False,
                    'fit_matches_recommendation': fit_matches,
                    'recommended_fit': recommended_fit,
                    'fit_message': f"This {product.category} in size {rec_size} will fit you great!",
                })

        matching_products.sort(key=lambda x: (
            not x['fit_matches_recommendation'],
            not x['is_perfect_match'],
            x['product'].name,
        ))
        return matching_products[:limit]

    # ── Generate & save Recommendation rows ───────────────────────
    def generate_recommendations_for_scan(self, body_scan) -> List[object]:
        from fitting_system.models import Recommendation

        measurements = {
            'height': float(body_scan.height),
            'chest': float(body_scan.chest),
            'waist': float(body_scan.waist),
            'shoulder_width': float(body_scan.shoulder_width),
        }
        if body_scan.hip:
            measurements['hip'] = float(body_scan.hip)
        if body_scan.torso_length:
            measurements['torso_length'] = float(body_scan.torso_length)
        if body_scan.arm_length:
            measurements['arm_length'] = float(body_scan.arm_length)
        if body_scan.inseam:
            measurements['inseam'] = float(body_scan.inseam)

        body_shape = getattr(body_scan, 'body_shape', 'rectangle') or 'rectangle'
        undertone = getattr(body_scan, 'undertone', 'warm')

        # Gemini
        base_recommended_size = self.recommend_size(measurements, body_shape=body_shape)
        recommended_fit = self.recommend_fit(measurements, body_shape=body_shape)
        color_rec = self.recommend_colors(body_scan.skin_tone, undertone)
        recommended_colors_str = f"{color_rec['recommended_shirt']}, {color_rec['recommended_pants']}"

        # Product recommendations across genders
        product_recommendations = []
        for gender in ['men', 'women', 'unisex']:
            recs = self._recommend_products(
                measurements, body_scan.skin_tone, undertone,
                gender=gender, body_shape=body_shape, limit=10,
            )
            product_recommendations.extend(recs)

        # Dedupe
        seen = set()
        unique = []
        for product, priority in product_recommendations:
            if product.id not in seen:
                seen.add(product.id)
                unique.append((product, priority))
        unique.sort(key=lambda x: x[1], reverse=True)

        # Create Recommendation objects
        recs_created = []
        for product, priority in unique[:10]:
            rec_size = self.recommend_size(
                measurements, garment_type=product.category, body_shape=body_shape,
            )
            rec = Recommendation.objects.create(
                body_scan=body_scan,
                product=product,
                recommended_size=rec_size,
                recommended_fit=recommended_fit,
                recommended_colors=recommended_colors_str,
                priority=priority,
            )
            recs_created.append(rec)
        return recs_created

    def _recommend_products(
        self, measurements, skin_tone, undertone,
        gender='unisex', body_shape='rectangle', limit=10,
    ) -> List[Tuple[object, int]]:
        from fitting_system.models import Product, ProductVariant, Color

        recommended_size = self.recommend_size(measurements, body_shape=body_shape)
        recommended_fit = self.recommend_fit(measurements, body_shape=body_shape)

        color_rec = self.recommend_colors(skin_tone, undertone)
        rec_shirt_color = Color.objects.filter(name=color_rec['recommended_shirt']).first()
        rec_pants_color = Color.objects.filter(name=color_rec['recommended_pants']).first()

        products = Product.objects.filter(
            Q(gender=gender) | Q(gender='unisex')
        )

        recommendations = []
        for product in products:
            available = ProductVariant.objects.filter(
                product=product, inventory__quantity__gt=0,
            )
            if not available.exists():
                continue

            priority = 5  # base

            if product.fit_type == recommended_fit:
                priority += 15

            if available.filter(size__name=recommended_size).exists():
                priority += 10

            is_top = product.category in ('shirt', 'jacket', 'dress')
            target = rec_shirt_color if is_top else rec_pants_color
            if target and available.filter(color=target).exists():
                priority += 10

            recommendations.append((product, priority))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]

    # _fallback_recommend_size REMOVED – Gemini AI is the sole source.
