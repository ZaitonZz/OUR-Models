from django.contrib import admin

from .models import ImageJob


@admin.register(ImageJob)
class ImageJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'image', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('image', 'error')
    readonly_fields = ('created_at', 'updated_at')

# Register your models here.
