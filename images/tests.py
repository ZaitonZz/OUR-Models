import shutil
import tempfile
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
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
        response = self.client.post(
            '/api/images/',
            {
                'external_id': 'website-1',
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_requires_external_id(self):
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_requires_callback_url(self):
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'external_id': 'website-1',
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_rejects_duplicate_external_id(self):
        ImageJob.objects.create(
            external_id='website-1',
            image=self.make_image_upload(),
            callback_url='https://example.com/api/results',
        )

        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-1',
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 409)

    @patch('images.services.requests.post')
    @patch('images.services.get_detector')
    @patch('images.services.DocumentPreprocessor.load_config')
    def test_upload_api_runs_inference_and_posts_callback(self, mock_load_config, mock_get_detector, mock_post):
        mock_load_config.return_value = Mock(run=Mock(return_value=self.make_preprocess_result()))
        mock_get_detector.return_value = Mock(
            predict=Mock(return_value=self.make_inference_result())
        )
        mock_post.return_value = Mock(status_code=200, text='ok')
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'external_id': 'website-1',
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])
        self.assertEqual(job.external_id, 'website-1')
        self.assertEqual(job.status, ImageJob.Status.COMPLETE)
        self.assertTrue(job.preprocessed_image.name)
        self.assertEqual(job.preprocessing['method'], 'brightness')
        self.assertEqual(job.preprocessing['skew_status'], 'flat')
        self.assertEqual(job.preprocessing['patch_counts']['body'], 2)
        self.assertEqual(job.result['label'], 'genuine')
        self.assertEqual(job.result['score'], 0.1234)
        self.assertIn('/media/preprocessed/', payload['preprocessed_image_url'])
        mock_get_detector.return_value.predict.assert_called_once()

        mock_post.assert_called_once()
        callback_url = mock_post.call_args.args[0]
        callback_payload = mock_post.call_args.kwargs['json']
        self.assertEqual(callback_url, 'https://example.com/api/results')
        self.assertEqual(callback_payload['external_id'], 'website-1')
        self.assertEqual(callback_payload['job_id'], payload['id'])
        self.assertEqual(callback_payload['status'], ImageJob.Status.COMPLETE)
        self.assertIn('/media/preprocessed/', callback_payload['preprocessed_image_url'])
        self.assertEqual(callback_payload['method'], 'brightness')
        self.assertEqual(callback_payload['patch_counts']['body'], 2)
        self.assertEqual(callback_payload['result']['label'], 'genuine')
        self.assertEqual(callback_payload['result']['top_roi'], 'footer')
        self.assertEqual(payload['callback_status_code'], 200)

    @patch('images.services.requests.post')
    @patch('images.services.get_detector')
    @patch('images.services.DocumentPreprocessor.load_config')
    def test_inference_failure_marks_failed_and_posts_callback(self, mock_load_config, mock_get_detector, mock_post):
        mock_load_config.return_value = Mock(run=Mock(return_value=self.make_preprocess_result()))
        mock_get_detector.return_value = Mock(
            predict=Mock(
                return_value=SimpleNamespace(
                    success=False,
                    label=None,
                    score=None,
                    roi_scores=None,
                    top_roi=None,
                    error='Empty patch list',
                )
            )
        )
        mock_post.return_value = Mock(status_code=200, text='ok')

        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-inference-fail',
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])
        self.assertEqual(job.status, ImageJob.Status.FAILED)
        self.assertEqual(job.result, {})
        callback_payload = mock_post.call_args.kwargs['json']
        self.assertEqual(callback_payload['status'], ImageJob.Status.FAILED)
        self.assertEqual(callback_payload['result'], {})
        self.assertEqual(callback_payload['error'], 'Empty patch list')

    @patch('images.services.requests.post')
    def test_invalid_image_marks_failed_and_posts_callback(self, mock_post):
        mock_post.return_value = Mock(status_code=200, text='ok')
        upload = SimpleUploadedFile('bad.txt', b'not an image', content_type='text/plain')

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'external_id': 'website-bad',
                'callback_url': 'https://example.com/api/results',
            },
        )

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])
        self.assertEqual(job.status, ImageJob.Status.FAILED)
        mock_post.assert_called_once()
        callback_payload = mock_post.call_args.kwargs['json']
        self.assertEqual(callback_payload['external_id'], 'website-bad')
        self.assertEqual(callback_payload['status'], ImageJob.Status.FAILED)
        self.assertEqual(callback_payload['preprocessed_image_url'], '')
        self.assertTrue(callback_payload['error'])
        self.assertEqual(payload['callback_status_code'], 200)

    def test_status_api_returns_external_id_and_preprocessing_metadata(self):
        job = ImageJob.objects.create(
            external_id='website-1',
            image=self.make_image_upload(),
            callback_url='https://example.com/api/results',
            status=ImageJob.Status.COMPLETE,
            preprocessing={
                'method': 'brightness',
                'skew_status': 'flat',
                'patch_counts': {'header': 1, 'body': 2, 'footer': 3},
            },
            result={
                'success': True,
                'label': 'genuine',
                'score': 0.1234,
                'roi_scores': {'header': 0.1, 'body': 0.12, 'footer': 0.15},
                'top_roi': 'footer',
                'error': '',
            },
        )

        response = self.client.get(f'/api/images/{job.pk}/')

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['external_id'], 'website-1')
        self.assertEqual(payload['preprocessing']['method'], 'brightness')
        self.assertEqual(payload['result']['label'], 'genuine')

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

    @staticmethod
    def make_preprocess_result():
        return SimpleNamespace(
            success=True,
            method='brightness',
            skew_status='flat',
            warped=np.full((1700, 1024, 3), 255, dtype=np.uint8),
            patches=[
                {
                    'array': np.full((128, 128, 3), 255, dtype=np.uint8),
                    'roi': 'footer',
                }
            ],
            patch_counts={'header': 1, 'body': 2, 'footer': 3},
            error=None,
        )

    @staticmethod
    def make_inference_result():
        return SimpleNamespace(
            success=True,
            label='genuine',
            score=0.1234,
            roi_scores={'header': 0.1, 'body': 0.12, 'footer': 0.15},
            top_roi='footer',
            error=None,
        )
