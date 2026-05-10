from pathlib import Path

import requests
from django.conf import settings
from PIL import Image, UnidentifiedImageError

from .models import ImageJob


def process_image_job(job: ImageJob, request=None) -> ImageJob:
    """Run image validation and model inference for an upload."""
    try:
        with Image.open(job.image.path) as image:
            image.verify()

        with Image.open(job.image.path) as image:
            width, height = image.size
            mode = image.mode
            format_name = image.format

        model_output = run_models(job.image.path)
        job.status = ImageJob.Status.COMPLETE
        job.result = {
            'file_name': Path(job.image.name).name,
            'width': width,
            'height': height,
            'mode': mode,
            'format': format_name,
            'models': model_output,
        }
        job.error = ''
    except (UnidentifiedImageError, OSError) as exc:
        job.status = ImageJob.Status.FAILED
        job.result = {}
        job.error = f'Invalid image file: {exc}'
    except Exception as exc:
        job.status = ImageJob.Status.FAILED
        job.result = {}
        job.error = str(exc)

    job.save(update_fields=['status', 'result', 'error', 'updated_at'])
    deliver_results(job, request=request)
    return job


def deliver_results(job: ImageJob, request=None) -> ImageJob:
    callback_url = job.callback_url or settings.RESULTS_API_URL
    if not callback_url:
        return job

    payload = build_result_payload(job, request=request)

    try:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=settings.RESULTS_API_TIMEOUT_SECONDS,
        )
        job.callback_status_code = response.status_code
        job.callback_response = response.text[:4000]
        job.callback_error = ''
    except requests.RequestException as exc:
        job.callback_status_code = None
        job.callback_response = ''
        job.callback_error = str(exc)

    job.save(
        update_fields=[
            'callback_status_code',
            'callback_response',
            'callback_error',
            'updated_at',
        ]
    )
    return job


def build_result_payload(job: ImageJob, request=None) -> dict:
    image_url = job.image.url
    if request is not None:
        image_url = request.build_absolute_uri(job.image.url)

    return {
        'id': job.pk,
        'status': job.status,
        'image_url': image_url,
        'result': job.result,
        'error': job.error,
    }


def run_models(image_path: str) -> dict:
    """
    Replace this with your real model pipeline.

    Keep the return value JSON-serializable so the result can be stored in the
    database and returned by the API.
    """
    return {
        'placeholder_model': {
            'label': 'not_configured',
            'confidence': 0.0,
            'source_path': image_path,
        }
    }
