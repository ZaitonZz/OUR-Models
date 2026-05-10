from pathlib import Path
from functools import lru_cache

import cv2
import requests
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils.text import get_valid_filename

from .models import ImageJob
from .inference import TORInference
from .ocr import extract_degree_from_image
from .preprocessing_pipeline import DocumentPreprocessor
from .signature_verification import verify_signatures


PREPROCESSOR_CONFIG_PATH = Path(__file__).with_name('tor_preprocessor_config.json')


def process_image_job(job: ImageJob, request=None, expected_signatures: dict | None = None) -> ImageJob:
    """Run TOR preprocessing and inference for an upload, then notify the website."""
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
        inference_result = run_models(preprocessing_result.patches)
        if not inference_result['success']:
            raise ValueError(inference_result.get('error') or 'Inference failed.')

        inference_result['signature_verification'] = verify_signatures(
            preprocessing_result.warped,
            external_id=job.external_id,
            request=request,
            expected_signatures=expected_signatures or {},
        )
        inference_result['degree_extraction'] = extract_degree_from_image(preprocessing_result.warped)

        job.status = ImageJob.Status.COMPLETE
        job.result = inference_result
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
        'result': job.result,
        'error': job.error,
    }


@lru_cache(maxsize=1)
def get_detector() -> TORInference:
    device = settings.TOR_INFERENCE_DEVICE or None
    return TORInference(
        weights_path=settings.TOR_MODEL_WEIGHTS_PATH,
        threshold=settings.TOR_INFERENCE_THRESHOLD,
        device=device,
    )


def run_models(patches: list) -> dict:
    inference_result = get_detector().predict(patches)
    return {
        'success': inference_result.success,
        'label': inference_result.label,
        'score': inference_result.score,
        'roi_scores': inference_result.roi_scores,
        'top_roi': inference_result.top_roi,
        'error': inference_result.error or '',
    }
