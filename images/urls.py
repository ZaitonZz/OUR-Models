from django.urls import path

from . import views

urlpatterns = [
    path('', views.upload_page, name='upload_page'),
    path('images/<int:pk>/', views.image_detail, name='image_detail'),
    path('api/images/', views.image_upload_api, name='image_upload_api'),
    path('api/images/<int:pk>/', views.image_status_api, name='image_status_api'),
    path('api/signature-references/sync/', views.signature_reference_sync_api, name='signature_reference_sync_api'),
]
