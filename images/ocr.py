import re
from typing import Any

import cv2
import numpy as np


DEGREE_LABEL_PATTERN = re.compile(
    r'degree\s*(?:/|\s)+title\s*(?:/|\s)+course\s*:?\s*(?P<inline>.*)$',
    re.IGNORECASE,
)
DEGREE_VALUE_PATTERN = re.compile(
    r'\b(?:Bachelor|Master|Doctor|Associate)\s+(?:of|in)\s+\S.*',
    re.IGNORECASE,
)
DEGREE_VALUE_STOP_PATTERN = re.compile(
    r'\s+(?:semester admitted|student no|student number|date graduated|date of graduation|school year|year level)\b.*$',
    re.IGNORECASE,
)
DEGREE_REGION_CROPS = (
    (0.10, 0.20, 0.35, 0.98),
    (0.10, 0.28, 0.35, 0.98),
    (0.06, 0.38, 0.35, 0.98),
    (0.06, 0.38, 0.04, 0.96),
)


def extract_degree_from_path(image_path: str) -> dict[str, Any]:
    """Extract the Degree/Title/Course value from the original uploaded TOR image."""
    document_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if document_image is None:
        return unavailable_result(f'OCR failed: could not read image at {image_path}.')

    return extract_degree_from_image(document_image)


def extract_degree_from_image(document_image: np.ndarray) -> dict[str, Any]:
    """Extract the Degree/Title/Course value from a TOR image."""
    try:
        import pytesseract
    except ImportError:
        return unavailable_result('Install pytesseract and the Tesseract OCR binary to enable degree extraction.')

    try:
        texts = []
        degree = ''
        for region in prepare_degree_regions(document_image):
            text = pytesseract.image_to_string(region, config='--psm 6')
            texts.append(text)
            degree = extract_degree_from_text(text)
            if degree:
                break
    except Exception as exc:
        return unavailable_result(f'OCR failed: {exc}')

    raw_text = '\n--- OCR attempt ---\n'.join(texts)

    return {
        'success': degree != '',
        'degree': degree or None,
        'title': degree or None,
        'course': degree or None,
        'program_match': None,
        'message': 'Degree extracted from TOR OCR.' if degree else 'Degree/Title/Course label was not found by OCR.',
        'raw_text': raw_text,
    }


def prepare_degree_region(document_image: np.ndarray) -> np.ndarray:
    """Crop and sharpen the upper TOR area where Degree/Title/Course appears."""
    return prepare_degree_regions(document_image)[0]


def prepare_degree_regions(document_image: np.ndarray) -> list[np.ndarray]:
    """Return likely Degree/Title/Course OCR regions, from strictest to broadest."""
    if document_image.size == 0:
        return [document_image]

    height, width = document_image.shape[:2]
    regions = []

    for top_ratio, bottom_ratio, left_ratio, right_ratio in DEGREE_REGION_CROPS:
        top = int(height * top_ratio)
        bottom = int(height * bottom_ratio)
        left = int(width * left_ratio)
        right = int(width * right_ratio)
        region = document_image[top:bottom, left:right]

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled, None, 12, 7, 21)
        regions.append(
            cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        )

    return regions


def extract_degree_from_text(text: str) -> str:
    lines = [normalize_ocr_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):
        match = DEGREE_LABEL_PATTERN.search(line)
        if not match:
            continue

        inline_value = clean_degree_value(match.group('inline'))
        if inline_value:
            return inline_value

        for next_line in lines[index + 1:]:
            value = clean_degree_value(next_line)
            if value:
                return value

    for line in lines:
        value = clean_degree_value(line)
        if value and DEGREE_VALUE_PATTERN.search(value):
            return value

    return ''


def normalize_ocr_line(line: str) -> str:
    return re.sub(r'\s+', ' ', line.replace('|', '/')).strip()


def clean_degree_value(value: str) -> str:
    degree_match = DEGREE_VALUE_PATTERN.search(value)
    if degree_match:
        value = degree_match.group(0)

    value = DEGREE_VALUE_STOP_PATTERN.sub('', value)
    value = re.sub(r'^[\s:.\-]+', '', value)
    value = re.sub(r'[\s\-_.]+$', '', value)
    value = re.sub(r'\s+', ' ', value).strip()

    return value if len(value) >= 8 else ''


def unavailable_result(message: str) -> dict[str, Any]:
    return {
        'success': False,
        'degree': None,
        'title': None,
        'course': None,
        'program_match': None,
        'message': message,
        'raw_text': '',
    }
