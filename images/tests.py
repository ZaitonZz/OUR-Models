import json
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image

from .models import ImageJob
from .ocr import extract_degree_from_image, extract_degree_from_path, extract_degree_from_text
from .signature_verification import (
    SiameseResNet18,
    distance_to_score,
    extract_signatures,
    reference_image_paths,
    score_to_verdict,
    candidate_presence_failure_to_payload,
    signature_presence_gate,
    suppress_printed_letters_from_tor_signature,
    verify_signatures,
)


TEST_MEDIA_ROOT = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT, TOR_SERVICE_TOKEN='test-token')
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
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_rejects_missing_service_token(self):
        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-1',
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_rejects_invalid_service_token(self):
        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-1',
            },
            HTTP_X_TOR_SERVICE_TOKEN='wrong-token',
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_requires_external_id(self):
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'callback_url': 'https://example.com/api/results',
            },
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)

    def test_upload_api_rejects_unknown_model_key(self):
        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-unknown-model',
                'model_key': 'unknown_model',
            },
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ImageJob.objects.count(), 0)
        self.assertIn('Unknown model_key', response.json()['error'])

    @patch('images.services.extract_degree_from_path')
    @patch('images.services.verify_signatures')
    @patch('images.services.get_detector')
    @patch('images.services.DocumentPreprocessor.load_config')
    def test_upload_api_runs_efficientnet_topk_inference(self, mock_load_config, mock_get_detector, mock_verify_signatures, mock_extract_degree):
        mock_load_config.return_value = Mock(run=Mock(return_value=self.make_preprocess_result()))
        mock_get_detector.return_value = Mock(
            predict=Mock(return_value=self.make_topk_inference_result())
        )
        mock_verify_signatures.return_value = self.make_signature_verification()
        mock_extract_degree.return_value = self.make_degree_extraction()

        response = self.client.post(
            '/api/images/',
            {
                'image': self.make_image_upload(),
                'external_id': 'website-topk-model',
                'model_key': 'efficientnet_b0_topk',
                'expected_signatures': json.dumps(self.expected_signatures()),
            },
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])

        self.assertEqual(job.model_key, 'efficientnet_b0_topk')
        self.assertEqual(job.result['model_key'], 'efficientnet_b0_topk')
        self.assertEqual(job.result['model_label'], 'EfficientNet-B0 top-k aggregation')
        self.assertEqual(job.result['model_threshold'], 0.8)
        self.assertEqual(job.result['threshold'], 0.8)
        self.assertEqual(job.result['aggregation'], 'topk_mean')
        self.assertEqual(job.result['top_roi_score'], 0.933)
        self.assertEqual(job.result['roi_scores']['footer']['top5_mean'], 0.933)
        self.assertEqual(payload['model_key'], 'efficientnet_b0_topk')
        self.assertEqual(payload['model_label'], 'EfficientNet-B0 top-k aggregation')
        self.assertEqual(payload['model_threshold'], 0.8)
        mock_get_detector.assert_called_once_with('efficientnet_b0_topk')

    @patch('images.services.requests.post')
    @patch('images.services.extract_degree_from_path')
    @patch('images.services.verify_signatures')
    @patch('images.services.get_detector')
    @patch('images.services.DocumentPreprocessor.load_config')
    def test_upload_api_allows_missing_callback_url(self, mock_load_config, mock_get_detector, mock_verify_signatures, mock_extract_degree, mock_post):
        mock_load_config.return_value = Mock(run=Mock(return_value=self.make_preprocess_result()))
        mock_get_detector.return_value = Mock(
            predict=Mock(return_value=self.make_inference_result())
        )
        mock_verify_signatures.return_value = self.make_signature_verification()
        mock_extract_degree.return_value = self.make_degree_extraction()
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'external_id': 'website-1',
                'expected_signatures': json.dumps(self.expected_signatures()),
            },
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(ImageJob.objects.get().callback_url, '')
        self.assertEqual(ImageJob.objects.get().model_key, 'efficientnet_b0')
        self.assertEqual(response.json()['model_key'], 'efficientnet_b0')
        mock_post.assert_not_called()

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
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 409)

    @patch('images.services.requests.post')
    @patch('images.services.extract_degree_from_path')
    @patch('images.services.verify_signatures')
    @patch('images.services.get_detector')
    @patch('images.services.DocumentPreprocessor.load_config')
    def test_upload_api_runs_inference_and_posts_callback(self, mock_load_config, mock_get_detector, mock_verify_signatures, mock_extract_degree, mock_post):
        events = []

        def extract_degree_side_effect(_image_path):
            events.append('ocr')

            return self.make_degree_extraction()

        def preprocess_side_effect(**_kwargs):
            self.assertEqual(events, ['ocr'])
            events.append('preprocess')

            return self.make_preprocess_result()

        mock_load_config.return_value = Mock(run=Mock(side_effect=preprocess_side_effect))
        mock_get_detector.return_value = Mock(
            predict=Mock(return_value=self.make_inference_result())
        )
        mock_verify_signatures.return_value = self.make_signature_verification()
        mock_extract_degree.side_effect = extract_degree_side_effect
        mock_post.return_value = Mock(status_code=200, text='ok')
        upload = self.make_image_upload()

        response = self.client.post(
            '/api/images/',
            {
                'image': upload,
                'external_id': 'website-1',
                'callback_url': 'https://example.com/api/results',
                'model_key': 'resnet50_mean',
                'expected_signatures': json.dumps(self.expected_signatures()),
            },
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        job = ImageJob.objects.get(pk=payload['id'])
        self.assertEqual(job.external_id, 'website-1')
        self.assertEqual(job.model_key, 'resnet50_mean')
        self.assertEqual(job.status, ImageJob.Status.COMPLETE)
        self.assertTrue(job.preprocessed_image.name)
        self.assertEqual(job.preprocessing['method'], 'brightness')
        self.assertEqual(job.preprocessing['skew_status'], 'flat')
        self.assertEqual(job.preprocessing['patch_counts']['body'], 2)
        self.assertEqual(job.result['label'], 'genuine')
        self.assertEqual(job.result['score'], 0.1234)
        self.assertEqual(job.result['model_key'], 'resnet50_mean')
        self.assertEqual(job.result['model_label'], 'ResNet50 mean aggregation')
        self.assertTrue(job.result['signature_verification']['success'])
        self.assertEqual(
            job.result['degree_extraction']['degree'],
            'Bachelor of Science in Information Technology',
        )
        self.assertEqual(
            job.result['signature_verification']['signatures'][0]['best_match_name'],
            'Judito T. Abadia',
        )
        self.assertIn('/media/preprocessed/', payload['preprocessed_image_url'])
        self.assertEqual(payload['model_key'], 'resnet50_mean')
        self.assertEqual(payload['model_label'], 'ResNet50 mean aggregation')
        self.assertEqual(payload['model_threshold'], 0.34)
        mock_get_detector.assert_called_once_with('resnet50_mean')
        mock_get_detector.return_value.predict.assert_called_once()
        mock_verify_signatures.assert_called_once()
        mock_extract_degree.assert_called_once_with(job.image.path)
        self.assertEqual(events, ['ocr', 'preprocess'])
        self.assertEqual(
            mock_verify_signatures.call_args.kwargs['expected_signatures'],
            self.expected_signatures(),
        )

        mock_post.assert_called_once()
        callback_url = mock_post.call_args.args[0]
        callback_payload = mock_post.call_args.kwargs['json']
        self.assertEqual(callback_url, 'https://example.com/api/results')
        self.assertEqual(callback_payload['external_id'], 'website-1')
        self.assertEqual(callback_payload['job_id'], payload['id'])
        self.assertEqual(callback_payload['status'], ImageJob.Status.COMPLETE)
        self.assertEqual(callback_payload['model_key'], 'resnet50_mean')
        self.assertEqual(callback_payload['model_label'], 'ResNet50 mean aggregation')
        self.assertIn('/media/preprocessed/', callback_payload['preprocessed_image_url'])
        self.assertEqual(callback_payload['method'], 'brightness')
        self.assertEqual(callback_payload['patch_counts']['body'], 2)
        self.assertEqual(callback_payload['result']['label'], 'genuine')
        self.assertEqual(callback_payload['result']['top_roi'], 'footer')
        self.assertTrue(callback_payload['result']['signature_verification']['success'])
        self.assertEqual(
            callback_payload['result']['degree_extraction']['degree'],
            'Bachelor of Science in Information Technology',
        )
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
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
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
            HTTP_X_TOR_SERVICE_TOKEN='test-token',
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
        self.assertEqual(payload['model_key'], 'efficientnet_b0')
        self.assertEqual(payload['model_label'], 'EfficientNet-B0 baseline')
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

    @staticmethod
    def make_topk_inference_result():
        return SimpleNamespace(
            success=True,
            label='fake',
            score=0.933,
            roi_scores={
                'header': {'n_patches': 1, 'mean': 0.2, 'max': 0.2, 'top5_mean': 0.2},
                'body': {'n_patches': 2, 'mean': 0.4, 'max': 0.45, 'top5_mean': 0.4},
                'footer': {'n_patches': 3, 'mean': 0.8, 'max': 0.98, 'top5_mean': 0.933},
            },
            top_roi='footer',
            top_roi_score=0.933,
            aggregation='topk_mean',
            threshold=0.8,
            error=None,
        )

    @staticmethod
    def expected_signatures():
        return {
            'sig1_prepared_by': 'abadia',
            'sig2_checked_by': 'arabejo',
            'sig3_certified_by': 'maniscan',
        }

    @staticmethod
    def make_signature_verification():
        return {
            'success': True,
            'threshold': 0.85,
            'expected_signatures': ImageUploadApiTests.expected_signatures(),
            'signatures': [
                {
                    'slot': 'sig1_prepared_by',
                    'label': 'Prepared By',
                    'best_match_id': 'abadia',
                    'best_match_name': 'Judito T. Abadia',
                    'distance': 0.42,
                    'score': 0.62,
                    'verdict': 'GENUINE',
                    'is_match': True,
                    'ink_pixels': 25,
                    'bbox_xywh': [1, 2, 3, 4],
                    'band_crop_url': 'http://testserver/media/signatures/job/sig1_prepared_by_band.png',
                    'ink_mask_url': 'http://testserver/media/signatures/job/sig1_prepared_by_ink_mask.png',
                    'error': '',
                }
            ],
            'error': '',
        }

    @staticmethod
    def make_degree_extraction():
        return {
            'success': True,
            'degree': 'Bachelor of Science in Information Technology',
            'title': 'Bachelor of Science in Information Technology',
            'course': 'Bachelor of Science in Information Technology',
            'program_match': None,
            'message': 'Degree extracted from TOR OCR.',
            'raw_text': 'Degree/Title/Course:\nBachelor of Science in Information Technology',
        }


class SignatureReferenceSyncApiTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.references_root = Path(self.temp_dir) / 'references'
        self.personnel_path = Path(self.temp_dir) / 'signature_personnel.json'
        self.personnel_path.write_text(
            json.dumps({
                'sig1_prepared_by': [],
                'sig2_checked_by': [],
                'sig3_certified_by': [],
            }),
            encoding='utf-8',
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_signature_reference_sync_rejects_invalid_service_token(self):
        with override_settings(
            TOR_SERVICE_TOKEN='test-token',
            TOR_SIGNATURE_REFERENCES_ROOT=str(self.references_root),
            TOR_SIGNATURE_PERSONNEL_PATH=str(self.personnel_path),
        ):
            response = self.client.post('/api/signature-references/sync/', {
                'slot': 'sig1_prepared_by',
                'personnel_id': 'abadia',
                'personnel_name': 'Judito T. Abadia',
                'images': [self.make_signature_upload()],
            })

        self.assertEqual(response.status_code, 403)

    def test_signature_reference_sync_writes_files_and_updates_personnel(self):
        with override_settings(
            TOR_SERVICE_TOKEN='test-token',
            TOR_SIGNATURE_REFERENCES_ROOT=str(self.references_root),
            TOR_SIGNATURE_PERSONNEL_PATH=str(self.personnel_path),
        ):
            response = self.client.post(
                '/api/signature-references/sync/',
                {
                    'slot': 'sig1_prepared_by',
                    'personnel_id': 'abadia',
                    'personnel_name': 'Judito T. Abadia',
                    'images': [self.make_signature_upload('sample.png')],
                },
                HTTP_X_TOR_SERVICE_TOKEN='test-token',
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            (self.references_root / 'sig1_prepared_by' / 'abadia' / 'genuine' / 'sample.png').exists()
        )

        personnel = json.loads(self.personnel_path.read_text(encoding='utf-8'))
        self.assertEqual(personnel['sig1_prepared_by'][0]['id'], 'abadia')
        self.assertEqual(personnel['sig1_prepared_by'][0]['name'], 'Judito T. Abadia')

    @staticmethod
    def make_signature_upload(name='signature.png'):
        return SimpleUploadedFile(name, b'signature-bytes', content_type='image/png')


class OcrExtractionTests(TestCase):
    @patch('images.ocr.extract_degree_from_image')
    @patch('images.ocr.cv2.imread')
    def test_degree_extraction_reads_original_image_path(self, mock_imread, mock_extract_degree):
        image = np.full((100, 80, 3), 255, dtype=np.uint8)
        mock_imread.return_value = image
        mock_extract_degree.return_value = {'degree': 'Bachelor of Science in Information Technology'}

        result = extract_degree_from_path('/tmp/original-upload.png')

        mock_imread.assert_called_once_with('/tmp/original-upload.png', cv2.IMREAD_COLOR)
        mock_extract_degree.assert_called_once_with(image)
        self.assertEqual(result['degree'], 'Bachelor of Science in Information Technology')

    def test_degree_extraction_reads_value_from_next_line(self):
        text = """
        Name: Juan Dela Cruz
        Degree/Title/Course:
        Bachelor of Science in Information Technology
        """

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Information Technology',
        )

    def test_degree_extraction_reads_inline_value(self):
        text = 'Degree/Title/Course: Bachelor of Science in Computer Science'

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Computer Science',
        )

    def test_degree_extraction_finds_label_after_ocr_prefix_noise(self):
        text = 'random OCR noise Degree/Title/Course: Bachelor of Science in Accountancy'

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Accountancy',
        )

    def test_degree_extraction_ignores_short_noise(self):
        text = """
        Degree/Title/Course:
        B.S._
        Bachelor of Science in Information Systems
        """

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Information Systems',
        )

    def test_degree_extraction_strips_trailing_noise(self):
        text = 'Degree/Title/Course: Bachelor of Science in Psychology ..__'

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Psychology',
        )

    def test_degree_extraction_trims_semester_text_from_noisy_next_line(self):
        text = """
        . re Semester Admitted Degree/Title/Course
        1ST SEMESTER, SY 2022-2023 Bachelor of Science in Information Technology
        """

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Information Technology',
        )

    def test_degree_extraction_reads_master_degree_next_to_semester_admitted(self):
        text = """
        Semester Admitted Degree Title/Course
        1ST SEMESTER, SY 2013-2014 MASTER OF SCIENCE IN BIOLOGY (MSBio)
        """

        self.assertEqual(
            extract_degree_from_text(text),
            'MASTER OF SCIENCE IN BIOLOGY (MSBio)',
        )

    def test_degree_extraction_falls_back_to_degree_phrase_when_label_is_missed(self):
        text = '1ST SEMESTER, SY 2022-2023 Bachelor of Science in Information Technology'

        self.assertEqual(
            extract_degree_from_text(text),
            'Bachelor of Science in Information Technology',
        )

    def test_degree_extraction_reads_master_degree_when_label_is_missed(self):
        text = '1ST SEMESTER, SY 2013-2014 MASTER OF SCIENCE IN BIOLOGY (MSBio)'

        self.assertEqual(
            extract_degree_from_text(text),
            'MASTER OF SCIENCE IN BIOLOGY (MSBio)',
        )

    def test_degree_extraction_tries_multiple_image_regions(self):
        pytesseract = SimpleNamespace(
            image_to_string=Mock(
                side_effect=[
                    'Degree/Title/Course\n.So',
                    'Degree/Title/Course\nBachelor of Science in Information Technology',
                    '',
                    '',
                ]
            )
        )

        with patch.dict(sys.modules, {'pytesseract': pytesseract}):
            result = extract_degree_from_image(np.full((100, 100, 3), 255, dtype=np.uint8))

        self.assertTrue(result['success'])
        self.assertEqual(result['degree'], 'Bachelor of Science in Information Technology')
        self.assertEqual(pytesseract.image_to_string.call_count, 2)


class SignatureVerificationTests(TestCase):
    def test_siamese_checkpoint_loads_when_present(self):
        checkpoint = Path(__file__).resolve().parent.parent / 'siamese_resnet18_finetuned_no_leakage.pth'
        if not checkpoint.exists():
            self.skipTest('Siamese checkpoint is not present.')

        model = SiameseResNet18()
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    @override_settings(
        TOR_SIGNATURE_REFERENCES_ROOT=str(Path(TEST_MEDIA_ROOT) / 'missing-references'),
        TOR_SIGNATURE_DISTANCE_THRESHOLD=0.85,
    )
    def test_missing_reference_folder_returns_signature_error(self):
        image = np.full((1700, 1024, 3), 255, dtype=np.uint8)
        result = verify_signatures(image, external_id='website-1')

        self.assertFalse(result['success'])
        self.assertEqual(result['signatures'], [])
        self.assertIn('Signature reference folder does not exist', result['error'])

    def test_signature_extraction_returns_three_slots(self):
        image = np.full((1700, 1024, 3), 255, dtype=np.uint8)
        signatures = extract_signatures(image)

        self.assertEqual(
            [signature.slot for signature in signatures],
            ['sig1_prepared_by', 'sig2_checked_by', 'sig3_certified_by'],
        )
        self.assertTrue(all(signature.band_crop.size > 0 for signature in signatures))
        self.assertTrue(all(signature.ink_mask.size > 0 for signature in signatures))

    def test_distance_threshold_maps_to_half_similarity_score(self):
        self.assertEqual(distance_to_score(distance=0.3771, decision_threshold=0.3771), 0.5)

    def test_signature_presence_gate_rejects_blank_candidate(self):
        mask = np.zeros((120, 320), dtype=np.uint8)
        result = signature_presence_gate(mask)

        self.assertFalse(result['passed'])
        self.assertIn('too little ink', result['reason'])

    def test_candidate_presence_failure_returns_invalid_payload(self):
        presence = signature_presence_gate(np.zeros((120, 320), dtype=np.uint8))
        payload = candidate_presence_failure_to_payload(presence, decision_threshold=0.3771)

        self.assertEqual(payload['verdict'], 'INVALID')
        self.assertEqual(payload['reason'], 'no_signature_detected')
        self.assertFalse(payload['signature_detected'])
        self.assertIsNone(payload['score'])
        self.assertFalse(payload['model_inference_ran'])

    def test_signature_presence_gate_accepts_signature_like_stroke(self):
        mask = np.zeros((120, 320), dtype=np.uint8)
        cv2.line(mask, (40, 60), (250, 35), 255, 3)
        cv2.line(mask, (80, 64), (190, 88), 255, 2)
        result = signature_presence_gate(mask)

        self.assertTrue(result['passed'])
        self.assertGreater(result['signature_like_components'], 0)

    def test_score_to_verdict_has_manual_review_band(self):
        self.assertEqual(score_to_verdict(0.70), 'GENUINE')
        self.assertEqual(score_to_verdict(0.20), 'SUSPICIOUS')
        self.assertEqual(score_to_verdict(0.45), 'NEEDS MANUAL REVIEW')

    def test_printed_letter_suppression_preserves_non_empty_signature(self):
        mask = np.zeros((120, 320), dtype=np.uint8)
        cv2.putText(mask, 'NAME', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.line(mask, (60, 48), (250, 32), 255, 3)

        cleaned = suppress_printed_letters_from_tor_signature(mask)

        self.assertGreater(np.count_nonzero(cleaned), 0)

    def test_reference_image_paths_prefers_genuine_subfolder(self):
        person_dir = Path(TEST_MEDIA_ROOT) / 'references' / 'sig1_prepared_by' / 'abadia'
        genuine_dir = person_dir / 'genuine'
        forged_dir = person_dir / 'forged'
        genuine_dir.mkdir(parents=True, exist_ok=True)
        forged_dir.mkdir(parents=True, exist_ok=True)
        (person_dir / 'direct.png').write_bytes(b'direct')
        (genuine_dir / 'sample.png').write_bytes(b'genuine')
        (forged_dir / 'sample.png').write_bytes(b'forged')

        paths = reference_image_paths(person_dir)

        self.assertEqual(paths, [genuine_dir / 'sample.png'])
