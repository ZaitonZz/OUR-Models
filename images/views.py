from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ImageJob
from .services import process_image_job


def upload_page(request):
    latest_jobs = ImageJob.objects.order_by('-created_at')[:10]

    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return render(
                request,
                'images/upload.html',
                {'latest_jobs': latest_jobs, 'error': 'Choose an image to upload.'},
                status=400,
            )

        callback_url = request.POST.get('callback_url', '').strip()
        job = ImageJob.objects.create(image=image_file, callback_url=callback_url)
        process_image_job(job, request=request)
        return redirect('image_detail', pk=job.pk)

    return render(request, 'images/upload.html', {'latest_jobs': latest_jobs})


def image_detail(request, pk):
    job = get_object_or_404(ImageJob, pk=pk)
    return render(request, 'images/detail.html', {'job': job})


@csrf_exempt
@require_http_methods(['POST'])
def image_upload_api(request):
    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({'error': 'Upload an image file using the "image" field.'}, status=400)

    callback_url = request.POST.get('callback_url', '').strip()
    job = ImageJob.objects.create(image=image_file, callback_url=callback_url)
    process_image_job(job, request=request)

    response = {
        'id': job.pk,
        'status': job.status,
        'image_url': request.build_absolute_uri(job.image.url),
        'result': job.result,
        'error': job.error,
        'callback_url': job.callback_url or settings.RESULTS_API_URL,
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
            'status': job.status,
            'image_url': request.build_absolute_uri(job.image.url),
            'result': job.result,
            'error': job.error,
            'callback_url': job.callback_url or settings.RESULTS_API_URL,
            'callback_status_code': job.callback_status_code,
            'callback_error': job.callback_error,
            'created_at': job.created_at.isoformat(),
            'updated_at': job.updated_at.isoformat(),
        }
    )

# Create your views here.
