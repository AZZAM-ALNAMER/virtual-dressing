import random
from django.core.management.base import BaseCommand
from fitting_system.models import Size, Color, Product, ProductVariant, Inventory
from fitting_system.color_palettes import get_all_unique_colors, SKIN_TONE_PALETTES


class Command(BaseCommand):
    help = 'Populate database with MVP data - minimal clothing sets for men and women'

    def handle(self, *args, **kwargs):
        self.stdout.write('Clearing existing products...')
        Inventory.objects.all().delete()
        ProductVariant.objects.all().delete()
        Product.objects.all().delete()
        Color.objects.all().delete()

        self.stdout.write('Populating database with MVP data...')

        # ── Sizes ─────────────────────────────────────────────────
        self.stdout.write('Ensuring sizes exist...')
        sizes_data = [
            {'name': 'S', 'chest_min': 85, 'chest_max': 92, 'waist_min': 70, 'waist_max': 77,
             'shoulder_min': 40, 'shoulder_max': 43, 'height_min': 160, 'height_max': 170},
            {'name': 'M', 'chest_min': 93, 'chest_max': 100, 'waist_min': 78, 'waist_max': 85,
             'shoulder_min': 44, 'shoulder_max': 47, 'height_min': 168, 'height_max': 178},
            {'name': 'L', 'chest_min': 101, 'chest_max': 108, 'waist_min': 86, 'waist_max': 93,
             'shoulder_min': 48, 'shoulder_max': 51, 'height_min': 175, 'height_max': 185},
            {'name': 'XL', 'chest_min': 109, 'chest_max': 116, 'waist_min': 94, 'waist_max': 101,
             'shoulder_min': 52, 'shoulder_max': 55, 'height_min': 180, 'height_max': 190},
        ]
        for sd in sizes_data:
            Size.objects.get_or_create(name=sd['name'], defaults=sd)

        # ── Colours from unified palette ──────────────────────────
        self.stdout.write('Creating unified product colors...')
        all_colors = get_all_unique_colors()

        # Determine which colour names are used for shirts vs pants
        shirt_color_names = set()
        pants_color_names = set()
        for palette in SKIN_TONE_PALETTES.values():
            for c in palette['shirts']:
                shirt_color_names.add(c['name'])
            for c in palette['pants']:
                pants_color_names.add(c['name'])

        colors_map = {}  # name → Color instance
        for c in all_colors:
            obj, _ = Color.objects.get_or_create(
                name=c['name'],
                defaults={'hex_code': c['hex'], 'category': 'neutral'},
            )
            colors_map[c['name']] = obj
        self.stdout.write(f'  🎨 {len(colors_map)} unique colors created')

        # ── Products ──────────────────────────────────────────────
        self.stdout.write('Creating MVP products with palette colors...')
        products_config = [
            # ============ MEN'S SHIRTS ============
            {'product': {'name': 'Slim Fit Cotton Shirt', 'category': 'shirt', 'fit_type': 'slim',
                         'gender': 'men', 'price': 54.99,
                         'description': 'Modern slim fit cotton shirt for a sleek, tailored look.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Classic Cotton Shirt', 'category': 'shirt', 'fit_type': 'regular',
                         'gender': 'men', 'price': 49.99,
                         'description': 'A timeless classic cotton shirt perfect for any occasion.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Relaxed Cotton Shirt', 'category': 'shirt', 'fit_type': 'oversize',
                         'gender': 'men', 'price': 52.99,
                         'description': 'Comfortable relaxed fit cotton shirt with extra room.'},
             'color_pool': 'shirts'},
            # ============ MEN'S PANTS ============
            {'product': {'name': 'Slim Fit Jeans', 'category': 'pants', 'fit_type': 'slim',
                         'gender': 'men', 'price': 84.99,
                         'description': 'Modern slim fit denim jeans with a streamlined silhouette.'},
             'color_pool': 'pants'},
            {'product': {'name': 'Casual Denim Jeans', 'category': 'pants', 'fit_type': 'regular',
                         'gender': 'men', 'price': 79.99,
                         'description': 'Comfortable denim jeans with a classic fit.'},
             'color_pool': 'pants'},
            {'product': {'name': 'Loose Fit Jeans', 'category': 'pants', 'fit_type': 'oversize',
                         'gender': 'men', 'price': 82.99,
                         'description': 'Relaxed loose fit jeans for maximum comfort.'},
             'color_pool': 'pants'},
            # ============ MEN'S JACKETS ============
            {'product': {'name': 'Fitted Leather Jacket', 'category': 'jacket', 'fit_type': 'slim',
                         'gender': 'men', 'price': 219.99,
                         'description': 'Sleek fitted leather jacket with a modern cut.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Leather Jacket', 'category': 'jacket', 'fit_type': 'regular',
                         'gender': 'men', 'price': 199.99,
                         'description': 'Premium leather jacket with a classic design.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Oversized Leather Jacket', 'category': 'jacket', 'fit_type': 'oversize',
                         'gender': 'men', 'price': 229.99,
                         'description': 'Bold oversized leather jacket for a modern streetwear look.'},
             'color_pool': 'shirts'},
            # ============ WOMEN'S BLOUSES ============
            {'product': {'name': 'Fitted Blouse', 'category': 'shirt', 'fit_type': 'slim',
                         'gender': 'women', 'price': 59.99,
                         'description': 'Elegant fitted blouse with a tailored silhouette.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Elegant Blouse', 'category': 'shirt', 'fit_type': 'regular',
                         'gender': 'women', 'price': 54.99,
                         'description': 'Sophisticated blouse with delicate details.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Oversized Blouse', 'category': 'shirt', 'fit_type': 'oversize',
                         'gender': 'women', 'price': 56.99,
                         'description': 'Flowy oversized blouse for effortless chic style.'},
             'color_pool': 'shirts'},
            # ============ WOMEN'S DRESSES ============
            {'product': {'name': 'Fitted Summer Dress', 'category': 'dress', 'fit_type': 'slim',
                         'gender': 'women', 'price': 94.99,
                         'description': 'Form-fitting summer dress that accentuates your silhouette.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Summer Dress', 'category': 'dress', 'fit_type': 'regular',
                         'gender': 'women', 'price': 89.99,
                         'description': 'Light and breezy summer dress perfect for warm weather.'},
             'color_pool': 'shirts'},
            {'product': {'name': 'Flowy Summer Dress', 'category': 'dress', 'fit_type': 'oversize',
                         'gender': 'women', 'price': 92.99,
                         'description': 'Airy flowy dress with a relaxed fit.'},
             'color_pool': 'shirts'},
            # ============ WOMEN'S TROUSERS ============
            {'product': {'name': 'Slim Trousers', 'category': 'pants', 'fit_type': 'slim',
                         'gender': 'women', 'price': 79.99,
                         'description': 'Tailored slim fit trousers for a sleek, professional look.'},
             'color_pool': 'pants'},
            {'product': {'name': 'High-Waist Trousers', 'category': 'pants', 'fit_type': 'regular',
                         'gender': 'women', 'price': 74.99,
                         'description': 'Flattering high-waist trousers with a comfortable fit.'},
             'color_pool': 'pants'},
            {'product': {'name': 'Wide-Leg Trousers', 'category': 'pants', 'fit_type': 'oversize',
                         'gender': 'women', 'price': 77.99,
                         'description': 'Trendy wide-leg trousers with a relaxed, flowing silhouette.'},
             'color_pool': 'pants'},
        ]

        sizes = Size.objects.all()

        for config in products_config:
            product_data = config['product']
            pool_key = config['color_pool']  # 'shirts' or 'pants'

            product, created = Product.objects.get_or_create(
                name=product_data['name'], defaults=product_data,
            )
            if created:
                self.stdout.write(f'  Created: {product.name}')

            # Collect ALL unique colour names used in the chosen pool across
            # every skin tone so the product is available in every palette.
            pool_colors = set()
            for palette in SKIN_TONE_PALETTES.values():
                for c in palette[pool_key]:
                    pool_colors.add(c['name'])

            counter = 1
            for size in sizes:
                for color_name in sorted(pool_colors):
                    color = colors_map.get(color_name)
                    if not color:
                        continue
                    sku = f"{product.id}-{size.name}-{color.id}-{counter}"
                    variant, v_created = ProductVariant.objects.get_or_create(
                        product=product, size=size, color=color,
                        defaults={'sku': sku},
                    )
                    counter += 1
                    if v_created:
                        Inventory.objects.create(
                            product_variant=variant,
                            quantity=random.randint(10, 25),
                            low_stock_threshold=5,
                        )

            variant_colors = product.variants.values_list('color__name', flat=True).distinct()
            self.stdout.write(f'    Colors: {", ".join(sorted(variant_colors))}')

        # ── Summary ───────────────────────────────────────────────
        self.stdout.write(self.style.SUCCESS('\n✅ Successfully populated MVP database!'))
        self.stdout.write(f'📦 Products:  {Product.objects.count()}')
        self.stdout.write(f'📏 Sizes:     {Size.objects.count()}')
        self.stdout.write(f'🎨 Colors:    {Color.objects.count()}')
        self.stdout.write(f'🏷️  Variants:  {ProductVariant.objects.count()}')
        self.stdout.write(f'📊 Inventory: {Inventory.objects.count()}')

        self.stdout.write('\n📋 MVP Product Summary:')
        self.stdout.write("  Men's Set:")
        for p in Product.objects.filter(gender='men').order_by('category', 'fit_type'):
            cl = p.variants.values_list('color__name', flat=True).distinct()
            self.stdout.write(f'    • {p.name} ({p.category} - {p.fit_type})')
            self.stdout.write(f'      Colors: {", ".join(sorted(cl))}')
        self.stdout.write("  Women's Set:")
        for p in Product.objects.filter(gender='women').order_by('category', 'fit_type'):
            cl = p.variants.values_list('color__name', flat=True).distinct()
            self.stdout.write(f'    • {p.name} ({p.category} - {p.fit_type})')
            self.stdout.write(f'      Colors: {", ".join(sorted(cl))}')
