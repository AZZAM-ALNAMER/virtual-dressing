from django.db import models
from django.utils import timezone
import uuid


class Size(models.Model):
    """Size definitions with measurement ranges"""
    name = models.CharField(max_length=10, unique=True)  # S, M, L, XL, XXL
    chest_min = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    chest_max = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    waist_min = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    waist_max = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    shoulder_min = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    shoulder_max = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    height_min = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    height_max = models.DecimalField(max_digits=5, decimal_places=2)  # cm
    
    class Meta:
        ordering = ['id']
    
    def __str__(self):
        return self.name


class Color(models.Model):
    """Color definitions with hex codes"""
    CATEGORY_CHOICES = [
        ('light', 'Light Colors'),
        ('medium', 'Medium Colors'),
        ('dark', 'Dark Colors'),
        ('neutral', 'Neutral Colors'),
        ('vibrant', 'Vibrant Colors'),
    ]
    
    name = models.CharField(max_length=50)
    hex_code = models.CharField(max_length=7)  # e.g., #FF5733
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Product(models.Model):
    """Clothing products"""
    CATEGORY_CHOICES = [
        ('shirt', 'Shirt'),
        ('pants', 'Pants'),
        ('jacket', 'Jacket'),
        ('dress', 'Dress'),
        ('skirt', 'Skirt'),
    ]
    
    FIT_TYPE_CHOICES = [
        ('slim', 'Slim Fit'),
        ('regular', 'Regular Fit'),
        ('oversize', 'Oversize Fit'),
    ]
    
    GENDER_CHOICES = [
        ('men', 'Men'),
        ('women', 'Women'),
        ('unisex', 'Unisex'),
    ]
    
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    fit_type = models.CharField(max_length=20, choices=FIT_TYPE_CHOICES)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    image_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.gender} - {self.fit_type})"


class ProductVariant(models.Model):
    """Combination of product + size + color"""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='variants')
    size = models.ForeignKey(Size, on_delete=models.CASCADE)
    color = models.ForeignKey(Color, on_delete=models.CASCADE)
    sku = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['product', 'size', 'color']
        unique_together = ['product', 'size', 'color']
    
    def __str__(self):
        return f"{self.product.name} - {self.size.name} - {self.color.name}"


class Inventory(models.Model):
    """Stock tracking for each variant"""
    product_variant = models.OneToOneField(ProductVariant, on_delete=models.CASCADE, related_name='inventory')
    quantity = models.IntegerField(default=0)
    low_stock_threshold = models.IntegerField(default=5)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = 'Inventories'
    
    def __str__(self):
        return f"{self.product_variant} - Stock: {self.quantity}"
    
    @property
    def is_low_stock(self):
        return 0 < self.quantity <= self.low_stock_threshold
    
    @property
    def is_out_of_stock(self):
        return self.quantity <= 0
    
    @property
    def is_available(self):
        return self.quantity > 0


class BodyScan(models.Model):
    """Stored measurement data (no images)"""
    SKIN_TONE_CHOICES = [
        ('very_light', 'Very Light'),
        ('light', 'Light'),
        ('intermediate', 'Intermediate'),
        ('tan', 'Tan'),
        ('dark', 'Dark'),
    ]
    
    UNDERTONE_CHOICES = [
        ('warm', 'Warm'),
        ('cool', 'Cool'),
    ]
    
    BODY_SHAPE_CHOICES = [
        ('hourglass', 'Hourglass'),
        ('rectangle', 'Rectangle'),
        ('triangle', 'Triangle (Pear)'),
        ('inverted_triangle', 'Inverted Triangle (Athletic)'),
        ('oval', 'Oval (Apple)'),
    ]
    
    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    
    # Core measurements (required)
    height = models.DecimalField(max_digits=5, decimal_places=1)  # cm - used for calibration
    shoulder_width = models.DecimalField(max_digits=5, decimal_places=1)  # cm
    chest = models.DecimalField(max_digits=5, decimal_places=1)  # cm - circumference
    waist = models.DecimalField(max_digits=5, decimal_places=1)  # cm - circumference
    
    # Fashion-specific measurements (nullable for backward compatibility)
    hip = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)  # cm - circumference
    torso_length = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)  # cm
    arm_length = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)  # cm
    inseam = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)  # cm - for pants
    
    # Body shape classification
    body_shape = models.CharField(max_length=20, choices=BODY_SHAPE_CHOICES, null=True, blank=True)
    
    # Skin analysis
    skin_tone = models.CharField(max_length=15, choices=SKIN_TONE_CHOICES)
    undertone = models.CharField(max_length=10, choices=UNDERTONE_CHOICES, default='warm')
    
    # Measurement quality metrics
    confidence_score = models.DecimalField(max_digits=3, decimal_places=2, default=1.0)  # 0.0-1.0
    frame_count = models.IntegerField(default=1)  # Number of frames used for averaging
    
    scanned_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-scanned_at']
    
    def __str__(self):
        return f"Scan {self.session_id} - {self.scanned_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def chest_to_waist_ratio(self):
        """Calculate body proportion ratio for fit recommendation"""
        if self.waist > 0:
            return float(self.chest) / float(self.waist)
        return 1.0
    
    @property
    def body_shape_display(self):
        """Get human-readable body shape name"""
        if self.body_shape:
            return dict(self.BODY_SHAPE_CHOICES).get(self.body_shape, self.body_shape)
        return None


class Recommendation(models.Model):
    """Generated recommendations"""
    body_scan = models.ForeignKey(BodyScan, on_delete=models.CASCADE, related_name='recommendations')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    recommended_size = models.CharField(max_length=10)
    recommended_fit = models.CharField(max_length=20)
    recommended_colors = models.TextField()  # Comma-separated color names
    priority = models.IntegerField(default=0)  # Higher = more relevant
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-priority', 'product']
    
    def __str__(self):
        return f"Recommendation for {self.body_scan.session_id} - {self.product.name}"
    
    def get_recommended_colors_list(self):
        """Return recommended colors as a list"""
        return [c.strip() for c in self.recommended_colors.split(',') if c.strip()]
