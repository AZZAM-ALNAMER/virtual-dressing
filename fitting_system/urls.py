from django.urls import path
from . import views

app_name = 'fitting_system'

urlpatterns = [
    path('', views.index, name='index'),
    path('scan/', views.scan, name='scan'),
    # API endpoints
    path('process-scan/', views.process_scan, name='process_scan'),
    path('process-scan-women/', views.process_scan_women, name='process_scan_women'),
    path('analyze-frame/', views.analyze_frame, name='analyze_frame'),
    path('recommendations/<uuid:session_id>/', views.recommendations, name='recommendations'),
    path('avatar/<uuid:session_id>/', views.avatar, name='avatar'),
    path('inventory/', views.inventory_dashboard, name='inventory'),
    path('api/inventory/', views.api_inventory, name='api_inventory'),
    path('store/', views.store, name='store'),
    path('store/product/<int:product_id>/', views.product_detail, name='product_detail'),
]
