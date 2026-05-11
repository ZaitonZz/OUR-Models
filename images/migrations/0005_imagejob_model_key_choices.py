from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('images', '0004_imagejob_model_key'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagejob',
            name='model_key',
            field=models.CharField(
                choices=[
                    ('efficientnet_b0', 'EfficientNet-B0 baseline'),
                    ('efficientnet_b0_topk', 'EfficientNet-B0 top-k aggregation'),
                    ('resnet50_mean', 'ResNet50 mean aggregation'),
                ],
                default='efficientnet_b0',
                max_length=64,
            ),
        ),
    ]
