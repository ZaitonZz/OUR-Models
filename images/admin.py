from django.contrib import admin

from .models import ImageJob


@admin.register(ImageJob)
class ImageJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'external_id', 'model_key', 'image', 'status', 'created_at', 'updated_at')
    list_filter = ('model_key', 'status', 'created_at')
    search_fields = ('external_id', 'image', 'error')
    readonly_fields = ('created_at', 'updated_at')

# Register your models here.
