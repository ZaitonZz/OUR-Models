from django.db import models


class ImageJob(models.Model):
    class ModelKey(models.TextChoices):
        EFFICIENTNET_B0 = 'efficientnet_b0', 'EfficientNet-B0 baseline'
        EFFICIENTNET_B0_TOPK = 'efficientnet_b0_topk', 'EfficientNet-B0 top-k aggregation'
        RESNET50_MEAN = 'resnet50_mean', 'ResNet50 mean aggregation'

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PREPROCESSING = 'preprocessing', 'Preprocessing'
        PREPROCESSED = 'preprocessed', 'Preprocessed'
        COMPLETE = 'complete', 'Complete'
        FAILED = 'failed', 'Failed'

    external_id = models.CharField(max_length=255, unique=True, db_index=True)
    model_key = models.CharField(
        max_length=64,
        choices=ModelKey.choices,
        default=ModelKey.EFFICIENTNET_B0,
    )
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
