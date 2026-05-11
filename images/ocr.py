import re
from typing import Any

import cv2
import numpy as np


DEGREE_LABEL_PATTERN = re.compile(
    r'degree\s*(?:/|\s)+title\s*(?:/|\s)+course\s*:?\s*(?P<inline>.*)$',
    re.IGNORECASE,
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
        text = pytesseract.image_to_string(prepare_degree_region(document_image), config='--psm 6')
    except Exception as exc:
        return unavailable_result(f'OCR failed: {exc}')

    degree = extract_degree_from_text(text)

    return {
        'success': degree != '',
        'degree': degree or None,
        'title': degree or None,
        'course': degree or None,
        'program_match': None,
        'message': 'Degree extracted from TOR OCR.' if degree else 'Degree/Title/Course label was not found by OCR.',
        'raw_text': text,
    }


def prepare_degree_region(document_image: np.ndarray) -> np.ndarray:
    """Crop and sharpen the upper TOR area where Degree/Title/Course appears."""
    if document_image.size == 0:
        return document_image

    height, width = document_image.shape[:2]
    top = int(height * 0.10)
    bottom = int(height * 0.20)
    left = int(width * 0.35)
    right = int(width * 0.98)
    region = document_image[top:bottom, left:right]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(scaled, None, 12, 7, 21)

    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


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

    return ''


def normalize_ocr_line(line: str) -> str:
    return re.sub(r'\s+', ' ', line.replace('|', '/')).strip()


def clean_degree_value(value: str) -> str:
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
