import json
from pathlib import Path

from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ImageJob
from .model_registry import DEFAULT_MODEL_KEY, get_model_config, model_metadata, normalize_model_key
from .services import process_image_job
from .signature_verification import get_reference_index


SIGNATURE_SLOTS = ('sig1_prepared_by', 'sig2_checked_by', 'sig3_certified_by')
VALID_SIGNATURE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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


def model_key_from_request(request):
    return normalize_model_key(request.POST.get('model_key', DEFAULT_MODEL_KEY))


def upload_page(request):
    latest_jobs = ImageJob.objects.order_by('-created_at')[:10]

    if request.method == 'POST':
        image_file = request.FILES.get('image')
        external_id = request.POST.get('external_id', '').strip()
        callback_url = request.POST.get('callback_url', '').strip()
        expected_signatures = expected_signatures_from_request(request)
        model_key = model_key_from_request(request)
        try:
            get_model_config(model_key)
        except ValueError as exc:
            return render(
                request,
                'images/upload.html',
                {
                    'latest_jobs': latest_jobs,
                    'error': str(exc),
                },
                status=400,
            )
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
            model_key=model_key,
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
    model_key = model_key_from_request(request)
    try:
        get_model_config(model_key)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
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
        model_key=model_key,
    )
    process_image_job(job, request=request, expected_signatures=expected_signatures)

    response = {
        'id': job.pk,
        'job_id': job.pk,
        'external_id': job.external_id,
        'status': job.status,
        **model_metadata(job.model_key),
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
            **model_metadata(job.model_key),
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


@csrf_exempt
@require_http_methods(['POST'])
def signature_reference_sync_api(request):
    if not is_authorized_service_request(request):
        return JsonResponse({'error': 'Invalid TOR service token.'}, status=403)

    slot = request.POST.get('slot', '').strip()
    personnel_id = request.POST.get('personnel_id', '').strip()
    personnel_name = request.POST.get('personnel_name', '').strip()
    images = request.FILES.getlist('images')

    if slot not in SIGNATURE_SLOTS:
        return JsonResponse({'error': 'Invalid signature slot.'}, status=422)

    if not personnel_id or not personnel_name:
        return JsonResponse({'error': 'Missing personnel_id or personnel_name.'}, status=422)

    if not images:
        return JsonResponse({'error': 'Upload at least one signature image.'}, status=422)

    target_dir = Path(settings.TOR_SIGNATURE_REFERENCES_ROOT) / slot / safe_path_segment(personnel_id) / 'genuine'
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_files = []

    for image in images:
        suffix = Path(image.name).suffix.lower()

        if suffix not in VALID_SIGNATURE_EXTENSIONS:
            return JsonResponse({'error': f'Unsupported signature image type: {image.name}'}, status=422)

        target_path = unique_reference_path(target_dir, image.name)

        with open(target_path, 'wb') as handle:
            for chunk in image.chunks():
                handle.write(chunk)

        stored_files.append(str(target_path))

    upsert_signature_personnel(slot, personnel_id, personnel_name)
    get_reference_index.cache_clear()

    return JsonResponse({
        'success': True,
        'slot': slot,
        'personnel_id': personnel_id,
        'stored_files': stored_files,
    })


def safe_path_segment(value):
    return ''.join(character if character.isalnum() or character in ('-', '_') else '_' for character in value)


def unique_reference_path(target_dir, original_name):
    safe_name = safe_path_segment(Path(original_name).stem) or 'signature'
    suffix = Path(original_name).suffix.lower()
    candidate = target_dir / f'{safe_name}{suffix}'
    counter = 2

    while candidate.exists():
        candidate = target_dir / f'{safe_name}-{counter}{suffix}'
        counter += 1

    return candidate


def upsert_signature_personnel(slot, personnel_id, personnel_name):
    path = Path(getattr(settings, 'TOR_SIGNATURE_PERSONNEL_PATH', Path(__file__).with_name('signature_personnel.json')))

    with open(path, 'r', encoding='utf-8') as handle:
        personnel = json.load(handle)

    people = personnel.setdefault(slot, [])
    existing = next((person for person in people if person.get('id') == personnel_id), None)

    if existing is None:
        people.append({'id': personnel_id, 'name': personnel_name})
    else:
        existing['name'] = personnel_name

    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(personnel, handle, indent=2)
        handle.write('\n')

# Create your views here.
