from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('images', '0003_preprocessing_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagejob',
            name='model_key',
            field=models.CharField(default='efficientnet_b0', max_length=64),
        ),
    ]
