from pathlib import Path

import cv2
import requests
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils.text import get_valid_filename

from .models import ImageJob
from .preprocessing_pipeline import DocumentPreprocessor


PREPROCESSOR_CONFIG_PATH = Path(__file__).with_name('tor_preprocessor_config.json')


def process_image_job(job: ImageJob, request=None) -> ImageJob:
    """Run TOR preprocessing for an upload and notify the website."""
    job.status = ImageJob.Status.PREPROCESSING
    job.error = ''
    job.save(update_fields=['status', 'error', 'updated_at'])

    try:
        preprocessor = DocumentPreprocessor.load_config(str(PREPROCESSOR_CONFIG_PATH))
        preprocessing_result = preprocessor.run(image_path=job.image.path)
        if not preprocessing_result.success:
            raise ValueError(preprocessing_result.error or 'Preprocessing failed.')

        save_preprocessed_image(job, preprocessing_result.warped)
        job.status = ImageJob.Status.PREPROCESSED
        job.preprocessing = {
            'method': preprocessing_result.method,
            'skew_status': preprocessing_result.skew_status,
            'patch_counts': preprocessing_result.patch_counts,
        }
        job.result = {}
        job.error = ''
    except Exception as exc:
        job.status = ImageJob.Status.FAILED
        job.preprocessing = {}
        job.result = {}
        job.error = str(exc)

    job.save(update_fields=['status', 'preprocessed_image', 'preprocessing', 'result', 'error', 'updated_at'])
    deliver_results(job, request=request)
    return job


def save_preprocessed_image(job: ImageJob, image_array) -> None:
    success, encoded_image = cv2.imencode('.jpg', image_array)
    if not success:
        raise ValueError('Could not encode preprocessed image.')

    original_name = Path(job.image.name).stem
    safe_external_id = get_valid_filename(job.external_id)
    safe_original_name = get_valid_filename(original_name)
    file_name = f'{safe_external_id}_{safe_original_name}_preprocessed.jpg'
    job.preprocessed_image.save(file_name, ContentFile(encoded_image.tobytes()), save=False)


def deliver_results(job: ImageJob, request=None) -> ImageJob:
    callback_url = job.callback_url
    if not callback_url:
        return job

    payload = build_result_payload(job, request=request)

    try:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=settings.PREPROCESSING_CALLBACK_TIMEOUT_SECONDS,
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
    preprocessed_image_url = job.preprocessed_image.url if job.preprocessed_image else ''
    if request is not None:
        image_url = request.build_absolute_uri(job.image.url)
        if preprocessed_image_url:
            preprocessed_image_url = request.build_absolute_uri(job.preprocessed_image.url)

    preprocessing = job.preprocessing or {}

    return {
        'external_id': job.external_id,
        'job_id': job.pk,
        'status': job.status,
        'original_image_url': image_url,
        'preprocessed_image_url': preprocessed_image_url,
        'method': preprocessing.get('method', ''),
        'skew_status': preprocessing.get('skew_status', ''),
        'patch_counts': preprocessing.get('patch_counts', {}),
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
