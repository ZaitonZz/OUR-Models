import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from django.conf import settings
from PIL import Image
from torchvision import transforms
from torchvision import models


SLOTS = {
    'sig1_prepared_by': 'Prepared By',
    'sig2_checked_by': 'Checked By',
    'sig3_certified_by': 'Certified By',
}
BASE_W = 1024
BASE_H = 1536
MANUAL_PIXEL_BBOXES_1024X1536 = {
    'sig1_prepared_by': (78, 1332, 250, 126),
    'sig2_checked_by': (420, 1322, 245, 158),
    'sig3_certified_by': (730, 1332, 255, 126),
}
SIGNATURE_BAND_FRAC = {
    'sig1_prepared_by': (0.00, 0.70),
    'sig2_checked_by': (0.00, 1.00),
    'sig3_certified_by': (0.00, 0.70),
}
SIGNATURE_CONFIG = {
    'safe_band_top_offset_px_1024x1536': 18,
    'safe_band_h_sig1_sig3_px_1024x1536': 95,
    'safe_band_h_sig2_px_1024x1536': 140,
    'safe_band_widen_px_1024x1536': 24,
    'fallback_signature_y_frac': 0.845,
}
CANVAS_W = 640
CANVAS_H = 360

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


@dataclass
class ExtractedSignature:
    slot: str
    label: str
    bbox_xywh: list[int]
    band_crop: np.ndarray
    ink_mask: np.ndarray
    ink_pixels: int


class SiameseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward_once(self, image):
        embedding = self.embedding_head(self.backbone(image))
        return nn.functional.normalize(embedding, p=2, dim=1)

    def forward(self, image_a, image_b):
        return self.forward_once(image_a), self.forward_once(image_b)


def load_personnel() -> dict:
    path = Path(__file__).with_name('signature_personnel.json')
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def verify_signatures(document_image: np.ndarray, external_id: str, request=None) -> dict:
    threshold = float(settings.TOR_SIGNATURE_DISTANCE_THRESHOLD)

    try:
        extracted = extract_signatures(document_image)
        reference_index = get_reference_index()
        signatures = [
            match_signature(signature, reference_index, threshold, external_id, request)
            for signature in extracted
        ]

        return {
            'success': True,
            'threshold': threshold,
            'signatures': signatures,
            'error': '',
        }
    except Exception as exc:
        return {
            'success': False,
            'threshold': threshold,
            'signatures': [],
            'error': str(exc),
        }


def extract_signatures(document_image: np.ndarray) -> list[ExtractedSignature]:
    bboxes, _remarks_y = find_signature_bboxes(document_image)
    signatures = []

    for slot, bbox in bboxes.items():
        x, y, w, h = bbox
        raw_roi = document_image[y:y + h, x:x + w]
        band_crop = crop_signature_band(raw_roi, slot)
        ink_mask = extract_signature_focused_mask(band_crop, slot)
        signatures.append(ExtractedSignature(
            slot=slot,
            label=SLOTS[slot],
            bbox_xywh=[int(x), int(y), int(w), int(h)],
            band_crop=band_crop,
            ink_mask=ink_mask,
            ink_pixels=int(np.count_nonzero(ink_mask)),
        ))

    return signatures


def match_signature(
    signature: ExtractedSignature,
    reference_index: dict,
    threshold: float,
    external_id: str,
    request=None,
) -> dict:
    allowed = {person['id']: person['name'] for person in load_personnel().get(signature.slot, [])}
    references = [
        reference for reference in reference_index.get(signature.slot, [])
        if reference['personnel_id'] in allowed
    ]

    band_url, mask_url = save_signature_artifacts(signature, external_id, request)

    if not references:
        return {
            'slot': signature.slot,
            'label': signature.label,
            'best_match_id': None,
            'best_match_name': None,
            'distance': None,
            'is_match': False,
            'ink_pixels': signature.ink_pixels,
            'bbox_xywh': signature.bbox_xywh,
            'band_crop_url': band_url,
            'ink_mask_url': mask_url,
            'error': f'No reference signatures found for {signature.label}.',
        }

    embedding = embed_image(signature.ink_mask)
    best = min(
        references,
        key=lambda reference: float(torch.dist(embedding, reference['embedding']).item()),
    )
    distance = float(torch.dist(embedding, best['embedding']).item())
    score = distance_to_score(distance, threshold)
    verdict = score_to_verdict(score)

    return {
        'slot': signature.slot,
        'label': signature.label,
        'best_match_id': best['personnel_id'],
        'best_match_name': allowed.get(best['personnel_id'], best['personnel_id']),
        'distance': round(distance, 4),
        'score': round(score, 4),
        'verdict': verdict,
        'is_match': verdict == 'GENUINE',
        'ink_pixels': signature.ink_pixels,
        'bbox_xywh': signature.bbox_xywh,
        'band_crop_url': band_url,
        'ink_mask_url': mask_url,
        'error': '',
    }


def save_signature_artifacts(signature: ExtractedSignature, external_id: str, request=None) -> tuple[str, str]:
    safe_external_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in external_id)
    relative_dir = Path('signatures') / safe_external_id
    output_dir = Path(settings.MEDIA_ROOT) / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    band_relative = relative_dir / f'{signature.slot}_band.png'
    mask_relative = relative_dir / f'{signature.slot}_ink_mask.png'
    cv2.imwrite(str(Path(settings.MEDIA_ROOT) / band_relative), signature.band_crop)
    cv2.imwrite(str(Path(settings.MEDIA_ROOT) / mask_relative), signature.ink_mask)

    band_url = f'{settings.MEDIA_URL}{band_relative.as_posix()}'
    mask_url = f'{settings.MEDIA_URL}{mask_relative.as_posix()}'

    if request is not None:
        return request.build_absolute_uri(band_url), request.build_absolute_uri(mask_url)

    return band_url, mask_url


@lru_cache(maxsize=1)
def get_signature_model() -> SiameseResNet18:
    device = torch.device(settings.TOR_SIGNATURE_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = SiameseResNet18().to(device)
    checkpoint = torch.load(settings.TOR_SIGNATURE_MODEL_WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_reference_index() -> dict:
    root = Path(settings.TOR_SIGNATURE_REFERENCES_ROOT)
    if not root.exists():
        raise FileNotFoundError(f'Signature reference folder does not exist: {root}')

    index = {slot: [] for slot in SLOTS}
    personnel = load_personnel()

    for slot, people in personnel.items():
        for person in people:
            person_dir = root / slot / person['id']
            if not person_dir.exists():
                continue

            for image_path in sorted(person_dir.glob('*')):
                if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    continue

                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                index.setdefault(slot, []).append({
                    'personnel_id': person['id'],
                    'path': str(image_path),
                    'embedding': embed_image(image),
                })

    return index


def embed_image(image: np.ndarray) -> torch.Tensor:
    if image is None or image.size == 0:
        raise ValueError('Cannot embed an empty signature image.')

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = to_white_ink_mask(image)
    normalized = center_on_16x9_canvas(mask)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    tensor = _transform(Image.fromarray(rgb)).unsqueeze(0)
    model = get_signature_model()
    device = next(model.parameters()).device

    with torch.no_grad():
        return model.forward_once(tensor.to(device)).cpu().squeeze(0)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith('module.') for key in state_dict):
        return state_dict

    return {
        key.replace('module.', '', 1): value
        for key, value in state_dict.items()
    }


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)

        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return strip_module_prefix(checkpoint)

    raise ValueError('Unsupported Siamese checkpoint format.')


def to_white_ink_mask(image: np.ndarray) -> np.ndarray:
    gray = image.copy()
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if float(np.mean(gray)) < 80:
        _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return remove_tiny_components(mask, min_area=18)


def center_on_16x9_canvas(mask: np.ndarray, margin: float = 0.14) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    if len(xs) == 0 or len(ys) == 0:
        return canvas

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    crop = mask[y1:y2 + 1, x1:x2 + 1]
    h, w = crop.shape[:2]

    if h <= 0 or w <= 0:
        return canvas

    max_w = int(CANVAS_W * (1.0 - 2.0 * margin))
    max_h = int(CANVAS_H * (1.0 - 2.0 * margin))
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (CANVAS_W - new_w) // 2
    y0 = (CANVAS_H - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def distance_to_score(distance: float, decision_threshold: float) -> float:
    if decision_threshold <= 0:
        raise ValueError('TOR_SIGNATURE_DISTANCE_THRESHOLD must be greater than 0.')

    score = 1.0 - (distance / (2.0 * decision_threshold))
    return float(np.clip(score, 0.0, 1.0))


def score_to_verdict(score: float) -> str:
    if score >= 0.60:
        return 'GENUINE'

    if score <= 0.35:
        return 'SUSPICIOUS'

    return 'NEEDS MANUAL REVIEW'


def scale_bbox_from_base(bbox, img_w, img_h):
    x, y, w, h = bbox
    sx = img_w / BASE_W
    sy = img_h / BASE_H
    x2 = max(0, min(int(round(x * sx)), img_w - 1))
    y2 = max(0, min(int(round(y * sy)), img_h - 1))
    w2 = max(1, min(int(round(w * sx)), img_w - x2))
    h2 = max(1, min(int(round(h * sy)), img_h - y2))
    return x2, y2, w2, h2


def detect_lower_remarks_line_y(tor_img: np.ndarray) -> Optional[int]:
    img_h, img_w = tor_img.shape[:2]
    gray = cv2.cvtColor(tor_img, cv2.COLOR_BGR2GRAY) if len(tor_img.shape) == 3 else tor_img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        13,
    )

    candidates = []
    for width_frac in (0.68, 0.55, 0.45, 0.35):
        kernel_w = max(70, int(img_w * width_frac))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        lines = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
        local = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if y < img_h * 0.55 or y > img_h * 0.92 or w < img_w * width_frac:
                continue
            if x < img_w * 0.22 or w > img_w * 0.65:
                local.append((int(y), int(area)))

        if local:
            candidates = local
            break

    if not candidates:
        return None

    ys = sorted(candidate[0] for candidate in candidates)
    groups = []
    for y in ys:
        if not groups or abs(groups[-1][-1] - y) > 8:
            groups.append([y])
        else:
            groups[-1].append(y)

    return max(int(round(sum(group) / len(group))) for group in groups)


def find_signature_bboxes(tor_img: np.ndarray) -> tuple[dict, Optional[int]]:
    img_h, img_w = tor_img.shape[:2]
    sx = img_w / BASE_W
    sy = img_h / BASE_H
    remarks_y = detect_lower_remarks_line_y(tor_img)
    base_y = (
        int(img_h * SIGNATURE_CONFIG['fallback_signature_y_frac'])
        if remarks_y is None
        else int(remarks_y + SIGNATURE_CONFIG['safe_band_top_offset_px_1024x1536'] * sy)
    )
    widen = int(round(SIGNATURE_CONFIG['safe_band_widen_px_1024x1536'] * sx))
    bboxes = {}

    for slot, base_bbox in MANUAL_PIXEL_BBOXES_1024X1536.items():
        x, _y, w, _h = scale_bbox_from_base(base_bbox, img_w, img_h)
        x1 = max(0, x - widen)
        x2 = min(img_w, x + w + widen)

        if slot == 'sig2_checked_by':
            h = int(round(SIGNATURE_CONFIG['safe_band_h_sig2_px_1024x1536'] * sy))
            y = base_y - int(round(8 * sy))
        else:
            h = int(round(SIGNATURE_CONFIG['safe_band_h_sig1_sig3_px_1024x1536'] * sy))
            y = base_y

        y = max(0, min(y, img_h - h))
        bboxes[slot] = (x1, y, x2 - x1, h)

    return bboxes, remarks_y


def crop_signature_band(raw_crop: np.ndarray, slot: str) -> np.ndarray:
    top_frac, bottom_frac = SIGNATURE_BAND_FRAC.get(slot, (0.0, 1.0))
    h = raw_crop.shape[0]
    y1 = max(0, min(int(round(h * top_frac)), h - 1))
    y2 = max(y1 + 1, min(int(round(h * bottom_frac)), h))
    return raw_crop[y1:y2, :]


def extract_signature_focused_mask(band_crop_bgr: np.ndarray, slot: str) -> np.ndarray:
    gray = cv2.cvtColor(band_crop_bgr, cv2.COLOR_BGR2GRAY) if len(band_crop_bgr.shape) == 3 else band_crop_bgr.copy()
    height, width = gray.shape[:2]
    denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    bg = cv2.GaussianBlur(denoised, (0, 0), sigmaX=max(9, width / 32), sigmaY=max(7, height / 10))
    flat = cv2.divide(denoised, bg, scale=185)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(flat, (3, 3), 0))
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9 if slot == 'sig2_checked_by' else 11,
    )
    otsu_thr, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cap = 142 if slot == 'sig2_checked_by' else 136
    dark_thr = int(np.clip(min(otsu_thr - 4, cap), 75, cap))
    dark_gate = (enhanced < dark_thr).astype(np.uint8) * 255
    ink = cv2.bitwise_and(cv2.bitwise_or(adaptive, otsu), dark_gate)
    ink = cv2.medianBlur(ink, 3)
    ink = remove_long_horizontal_lines(ink)
    ink = remove_tiny_components(ink, min_area=max(3, int(height * width * 0.000035)))
    ink = keep_signature_like_components(ink, slot)
    ink = remove_long_horizontal_lines(ink)
    ink = remove_tiny_components(ink, min_area=3)
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)


def remove_tiny_components(mask: np.ndarray, min_area=5) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == i] = 255
    return output


def remove_long_horizontal_lines(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(width * 0.32)), 1))
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    output = cv2.bitwise_and(mask, cv2.bitwise_not(lines))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(output, 8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        fill = area / max(w * h, 1)
        cy = y + h / 2
        if (
            w > width * 0.82 and h <= max(4, height * 0.035) and fill > 0.25
        ) or (
            w > width * 0.65 and h <= max(3, height * 0.025) and (cy < height * 0.12 or cy > height * 0.88)
        ):
            output[labels == i] = 0
    return output


def keep_signature_like_components(ink: np.ndarray, slot: str) -> np.ndarray:
    height, width = ink.shape[:2]
    output = np.zeros_like(ink)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    components = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 3:
            continue

        cx, cy = centroids[i]
        fill = area / max(w * h, 1)
        rel_area = area / max(height * width, 1)
        y_limit = 0.55 if slot == 'sig2_checked_by' else 0.50

        if cy > height * y_limit:
            continue
        if h < height * 0.20 and w < width * 0.16 and fill > 0.28 and area < height * width * 0.012:
            continue
        if w / max(h, 1) > 2.8 and h < height * 0.23 and fill > 0.18 and cy > height * 0.18:
            continue

        keep = (
            h >= height * 0.20
            or w >= width * 0.16
            or (w >= width * 0.08 and fill < 0.22)
            or (slot == 'sig2_checked_by' and h >= height * 0.32 and w <= width * 0.20)
            or (rel_area >= 0.0012 and fill < 0.45)
        )

        if keep:
            score = area + (1.8 * w) + (2.2 * h) - (70.0 * fill)
            components.append((score, i, x, y, w, h, area))

    if not components:
        return output

    components.sort(reverse=True, key=lambda item: item[0])
    best_score = components[0][0]
    kept = [component for component in components if component[0] >= best_score * 0.38]

    largest = max(kept, key=lambda item: item[6])
    _score, _i, lx, ly, lw, lh, _area = largest
    x1 = max(0, lx - int(width * 0.22))
    x2 = min(width, lx + lw + int(width * 0.22))
    y1 = max(0, ly - int(height * 0.25))
    y2 = height if slot == 'sig2_checked_by' else min(height, ly + lh + int(height * 0.30))

    for _score, i, x, y, w, h, _area in kept:
        if x <= x2 and x + w >= x1 and y <= y2 and y + h >= y1:
            output[labels == i] = 255

    return output
