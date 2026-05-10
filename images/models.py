from django.db import models


class ImageJob(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PREPROCESSING = 'preprocessing', 'Preprocessing'
        PREPROCESSED = 'preprocessed', 'Preprocessed'
        COMPLETE = 'complete', 'Complete'
        FAILED = 'failed', 'Failed'

    external_id = models.CharField(max_length=255, unique=True, db_index=True)
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    preprocessed_image = models.ImageField(upload_to='preprocessed/%Y/%m/%d/', blank=True)
    callback_url = models.URLField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    preprocessing = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    callback_status_code = models.PositiveIntegerField(null=True, blank=True)
    callback_response = models.TextField(blank=True)
    callback_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'ImageJob #{self.pk} ({self.status})'

# Create your models here.
