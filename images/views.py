import json

from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ImageJob
from .services import process_image_job


SIGNATURE_SLOTS = ('sig1_prepared_by', 'sig2_checked_by', 'sig3_certified_by')


def is_authorized_service_request(request):
    return request.headers.get('X-TOR-Service-Token') == settings.TOR_SERVICE_TOKEN


def expected_signatures_from_request(request):
    raw_expected_signatures = request.POST.get('expected_signatures', '').strip()
    if raw_expected_signatures:
        try:
            parsed = json.loads(raw_expected_signatures)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, dict):
            return {
                slot: value.strip()
                for slot, value in parsed.items()
                if slot in SIGNATURE_SLOTS and isinstance(value, str) and value.strip()
            }

    return {
        slot: value.strip()
        for slot in SIGNATURE_SLOTS
        for value in [request.POST.get(f'expected_signatures[{slot}]', '')]
        if value.strip()
    }


def upload_page(request):
    latest_jobs = ImageJob.objects.order_by('-created_at')[:10]

    if request.method == 'POST':
        image_file = request.FILES.get('image')
        external_id = request.POST.get('external_id', '').strip()
        callback_url = request.POST.get('callback_url', '').strip()
        expected_signatures = expected_signatures_from_request(request)
        if not image_file or not external_id or not callback_url:
            return render(
                request,
                'images/upload.html',
                {
                    'latest_jobs': latest_jobs,
                    'error': 'Choose an image and provide external_id plus callback_url.',
                },
                status=400,
            )
        if ImageJob.objects.filter(external_id=external_id).exists():
            return render(
                request,
                'images/upload.html',
                {
                    'latest_jobs': latest_jobs,
                    'error': 'A job already exists for that external_id.',
                },
                status=409,
            )

        job = ImageJob.objects.create(
            image=image_file,
            external_id=external_id,
            callback_url=callback_url,
        )
        process_image_job(job, request=request, expected_signatures=expected_signatures)
        return redirect('image_detail', pk=job.pk)

    return render(request, 'images/upload.html', {'latest_jobs': latest_jobs})


def image_detail(request, pk):
    job = get_object_or_404(ImageJob, pk=pk)
    return render(request, 'images/detail.html', {'job': job})


@csrf_exempt
@require_http_methods(['POST'])
def image_upload_api(request):
    if not is_authorized_service_request(request):
        return JsonResponse({'error': 'Invalid TOR service token.'}, status=403)

    image_file = request.FILES.get('image')
    external_id = request.POST.get('external_id', '').strip()
    callback_url = request.POST.get('callback_url', '').strip()
    expected_signatures = expected_signatures_from_request(request)
    missing_fields = [
        field_name
        for field_name, value in {
            'image': image_file,
            'external_id': external_id,
        }.items()
        if not value
    ]
    if missing_fields:
        return JsonResponse(
            {'error': f'Missing required field(s): {", ".join(missing_fields)}.'},
            status=400,
        )
    if ImageJob.objects.filter(external_id=external_id).exists():
        return JsonResponse(
            {'error': 'A job already exists for that external_id.'},
            status=409,
        )

    job = ImageJob.objects.create(
        image=image_file,
        external_id=external_id,
        callback_url=callback_url,
    )
    process_image_job(job, request=request, expected_signatures=expected_signatures)

    response = {
        'id': job.pk,
        'job_id': job.pk,
        'external_id': job.external_id,
        'status': job.status,
        'original_image_url': request.build_absolute_uri(job.image.url),
        'preprocessed_image_url': (
            request.build_absolute_uri(job.preprocessed_image.url)
            if job.preprocessed_image
            else ''
        ),
        'preprocessing': job.preprocessing,
        'result': job.result,
        'error': job.error,
        'callback_url': job.callback_url,
        'callback_status_code': job.callback_status_code,
        'callback_error': job.callback_error,
    }
    status = 201 if job.status == ImageJob.Status.COMPLETE else 422
    return JsonResponse(response, status=status)


def image_status_api(request, pk):
    job = get_object_or_404(ImageJob, pk=pk)
    return JsonResponse(
        {
            'id': job.pk,
            'job_id': job.pk,
            'external_id': job.external_id,
            'status': job.status,
            'original_image_url': request.build_absolute_uri(job.image.url),
            'preprocessed_image_url': (
                request.build_absolute_uri(job.preprocessed_image.url)
                if job.preprocessed_image
                else ''
            ),
            'preprocessing': job.preprocessing,
            'result': job.result,
            'error': job.error,
            'callback_url': job.callback_url,
            'callback_status_code': job.callback_status_code,
            'callback_error': job.callback_error,
            'created_at': job.created_at.isoformat(),
            'updated_at': job.updated_at.isoformat(),
        }
    )

# Create your views here.
