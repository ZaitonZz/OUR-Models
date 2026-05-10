import shutil
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image

from .models import ImageJob


TEST_MEDIA_ROOT = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class ImageUploadApiTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEST_MEDIA_ROOT, ignore_errors=True)

    def test_upload_api_requires_image_file(self):
        response = self.client.post('/api/images/', {})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_processes_valid_image(self):
        upload = self.make_image_upload()

        response = self.client.post('/api/images/', {'image': upload})

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])
        self.assertEqual(job.status, ImageJob.Status.COMPLETE)
        self.assertEqual(payload['result']['width'], 10)
        self.assertEqual(payload['result']['height'], 6)
        self.assertEqual(payload['result']['models']['placeholder_model']['label'], 'not_configured')

    @patch('images.services.requests.post')
    def test_upload_api_posts_results_to_callback_url(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text='ok')
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        mock_post.assert_called_once()
        callback_url = mock_post.call_args.args[0]
        callback_payload = mock_post.call_args.kwargs['json']
        self.assertEqual(callback_url, 'https://example.com/api/results')
        self.assertEqual(callback_payload['id'], payload['id'])
        self.assertEqual(callback_payload['status'], ImageJob.Status.COMPLETE)
        self.assertEqual(callback_payload['result']['width'], 10)
        self.assertEqual(payload['callback_status_code'], 200)

    @staticmethod
    def make_image_upload():
        image_bytes = BytesIO()
        image = Image.new('RGB', (10, 6), color='white')
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        return SimpleUploadedFile(
            'sample.png',
            image_bytes.read(),
            content_type='image/png',
        )
