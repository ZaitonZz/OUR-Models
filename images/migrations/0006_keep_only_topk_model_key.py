from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('images', '0005_imagejob_model_key_choices'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagejob',
            name='model_key',
            field=models.CharField(
                choices=[
                    ('efficientnet_b0_topk', 'EfficientNet-B0 top-k aggregation'),
                ],
                default='efficientnet_b0_topk',
                max_length=64,
            ),
        ),
    ]
