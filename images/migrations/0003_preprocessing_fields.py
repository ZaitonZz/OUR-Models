# Generated for TOR preprocessing integration.

import uuid

from django.db import migrations, models


def populate_external_ids(apps, schema_editor):
    ImageJob = apps.get_model('images', 'ImageJob')
    for job in ImageJob.objects.filter(external_id__isnull=True):
        job.external_id = f'legacy-{job.pk}-{uuid.uuid4().hex[:8]}'
        job.save(update_fields=['external_id'])


class Migration(migrations.Migration):

    dependencies = [
        ('images', '0002_imagejob_callback_error_imagejob_callback_response_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagejob',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending'),
                    ('preprocessing', 'Preprocessing'),
                    ('preprocessed', 'Preprocessed'),
                    ('complete', 'Complete'),
                    ('failed', 'Failed'),
                ],
                default='pending',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='imagejob',
            name='external_id',
            field=models.CharField(blank=True, db_index=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='imagejob',
            name='preprocessed_image',
            field=models.ImageField(blank=True, upload_to='preprocessed/%Y/%m/%d/'),
        ),
        migrations.AddField(
            model_name='imagejob',
            name='preprocessing',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.RunPython(populate_external_ids, migrations.RunPython.noop),
        migrations.AlterField(
            model_name='imagejob',
            name='external_id',
            field=models.CharField(db_index=True, max_length=255, unique=True),
        ),
    ]
