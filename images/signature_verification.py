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
CANVAS_FIT_SCALE = 0.78
CANVAS_PAD_RATIO = 0.08
GENUINE_SCORE_THRESHOLD = 0.60
SUSPICIOUS_SCORE_THRESHOLD = 0.35
PRESENCE_MIN_INK_PIXELS = 32
PRESENCE_MIN_COMPONENT_AREA = 16
PRESENCE_MIN_INK_RATIO = 0.0008
PRESENCE_MAX_INK_RATIO = 0.35

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


def verify_signatures(
    document_image: np.ndarray,
    external_id: str,
    request=None,
    expected_signatures: Optional[dict[str, str]] = None,
) -> dict:
    threshold = float(settings.TOR_SIGNATURE_DISTANCE_THRESHOLD)
    expected_signatures = expected_signatures or {}

    try:
        extracted = extract_signatures(document_image)
        reference_index = get_reference_index()
        signatures = [
            match_signature(
                signature,
                reference_index,
                threshold,
                external_id,
                request,
                expected_signatures=expected_signatures,
            )
            for signature in extracted
        ]

        return {
            'success': True,
            'threshold': threshold,
            'expected_signatures': expected_signatures,
            'signatures': signatures,
            'error': '',
        }
    except Exception as exc:
        return {
            'success': False,
            'threshold': threshold,
            'expected_signatures': expected_signatures,
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
    expected_signatures: Optional[dict[str, str]] = None,
) -> dict:
    allowed = {person['id']: person['name'] for person in load_personnel().get(signature.slot, [])}
    expected_signatures = expected_signatures or {}
    expected_id = expected_signatures.get(signature.slot)
    expected_name = allowed.get(expected_id, expected_id) if expected_id else None
    references = [
        reference for reference in reference_index.get(signature.slot, [])
        if reference['personnel_id'] in allowed
    ]

    if expected_id:
        references = [
            reference for reference in references
            if reference['personnel_id'] == expected_id
        ]

    candidate_pack = preprocess_signature_for_model(
        signature.ink_mask,
        is_tor_extracted=True,
        apply_16x9=True,
        suppress_printed_letters=True,
    )
    signature.ink_mask = candidate_pack['processed_mask']
    signature.ink_pixels = int(candidate_pack['presence']['ink_pixels'])
    band_url, mask_url = save_signature_artifacts(signature, external_id, request)

    if not references:
        return {
            'slot': signature.slot,
            'label': signature.label,
            'expected_match_id': expected_id,
            'expected_match_name': expected_name,
            'best_match_id': None,
            'best_match_name': None,
            'distance': None,
            'score': None,
            'verdict': None,
            'is_match': False,
            'model_inference_ran': False,
            'presence': candidate_pack['presence'],
            'preprocess': candidate_pack['preprocess'],
            'ink_pixels': signature.ink_pixels,
            'bbox_xywh': signature.bbox_xywh,
            'band_crop_url': band_url,
            'ink_mask_url': mask_url,
            'error': (
                f'No reference signatures found for expected {signature.label} signer.'
                if expected_id
                else f'No reference signatures found for {signature.label}.'
            ),
        }

    if not candidate_pack['presence']['passed']:
        invalid_payload = candidate_presence_failure_to_payload(candidate_pack['presence'], threshold)

        return {
            'slot': signature.slot,
            'label': signature.label,
            'expected_match_id': expected_id,
            'expected_match_name': expected_name,
            'best_match_id': None,
            'best_match_name': None,
            'distance': invalid_payload['distance'],
            'score': invalid_payload['score'],
            'verdict': invalid_payload['verdict'],
            'status': invalid_payload['status'],
            'reason': invalid_payload['reason'],
            'message': invalid_payload['message'],
            'signature_detected': invalid_payload['signature_detected'],
            'is_match': False,
            'model_inference_ran': invalid_payload['model_inference_ran'],
            'presence': candidate_pack['presence'],
            'presence_failure_detail': invalid_payload['presence_failure_detail'],
            'preprocess': candidate_pack['preprocess'],
            'ink_pixels': signature.ink_pixels,
            'bbox_xywh': signature.bbox_xywh,
            'band_crop_url': band_url,
            'ink_mask_url': mask_url,
            'error': invalid_payload['message'],
        }

    embedding = embed_preprocessed_signature(candidate_pack['model_mask'])
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
        'expected_match_id': expected_id,
        'expected_match_name': expected_name,
        'best_match_id': best['personnel_id'],
        'best_match_name': allowed.get(best['personnel_id'], best['personnel_id']),
        'distance': round(distance, 4),
        'score': round(score, 4),
        'verdict': verdict,
        'is_match': verdict == 'GENUINE',
        'model_inference_ran': True,
        'signature_detected': True,
        'presence': candidate_pack['presence'],
        'preprocess': candidate_pack['preprocess'],
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

            for image_path in reference_image_paths(person_dir):
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                index.setdefault(slot, []).append({
                    'personnel_id': person['id'],
                    'path': str(image_path),
                    'embedding': embed_image(image),
                })

    return index


def reference_image_paths(person_dir: Path) -> list[Path]:
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    search_dir = person_dir / 'genuine' if (person_dir / 'genuine').exists() else person_dir

    return [
        path for path in sorted(search_dir.glob('*'))
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]


def embed_image(image: np.ndarray) -> torch.Tensor:
    if image is None or image.size == 0:
        raise ValueError('Cannot embed an empty signature image.')

    pack = preprocess_signature_for_model(
        image,
        is_tor_extracted=False,
        apply_16x9=False,
        suppress_printed_letters=False,
    )

    return embed_preprocessed_signature(pack['model_mask'])


def embed_preprocessed_signature(model_mask: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(model_mask, cv2.COLOR_GRAY2RGB)
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
        if gray.shape[2] == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if float(np.mean(gray)) < 80:
        _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return remove_tiny_components(mask, min_area=5)


def signature_presence_gate(
    mask: np.ndarray,
    min_ink_pixels: int = PRESENCE_MIN_INK_PIXELS,
    min_component_area: int = PRESENCE_MIN_COMPONENT_AREA,
    min_ink_ratio: float = PRESENCE_MIN_INK_RATIO,
    max_ink_ratio: float = PRESENCE_MAX_INK_RATIO,
) -> dict:
    if mask is None or mask.size == 0:
        return {
            'passed': False,
            'reason': 'empty mask',
            'ink_pixels': 0,
            'ink_ratio': 0.0,
            'max_component_area': 0,
            'signature_like_components': 0,
        }

    binary = (mask > 0).astype(np.uint8) * 255
    height, width = binary.shape[:2]
    total_pixels = max(1, height * width)
    ink_pixels = int(np.count_nonzero(binary))
    ink_ratio = float(ink_pixels / total_pixels)

    if ink_pixels < min_ink_pixels:
        return {
            'passed': False,
            'reason': f'too little ink ({ink_pixels}px)',
            'ink_pixels': ink_pixels,
            'ink_ratio': round(ink_ratio, 6),
            'max_component_area': 0,
            'signature_like_components': 0,
        }

    if ink_ratio < min_ink_ratio:
        return {
            'passed': False,
            'reason': f'ink ratio too low ({ink_ratio:.6f})',
            'ink_pixels': ink_pixels,
            'ink_ratio': round(ink_ratio, 6),
            'max_component_area': 0,
            'signature_like_components': 0,
        }

    if ink_ratio > max_ink_ratio:
        return {
            'passed': False,
            'reason': f'ink ratio too high / noisy text region ({ink_ratio:.6f})',
            'ink_pixels': ink_pixels,
            'ink_ratio': round(ink_ratio, 6),
            'max_component_area': 0,
            'signature_like_components': 0,
        }

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    max_area = 0
    signature_like_components = 0

    for label_id in range(1, num_labels):
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        max_area = max(max_area, area)
        fill = area / float(max(1, comp_w * comp_h))
        long_or_tall = (comp_w >= width * 0.08) or (comp_h >= height * 0.08)
        sparse_stroke = fill < 0.70

        if area >= min_component_area and long_or_tall and sparse_stroke:
            signature_like_components += 1

    if max_area < min_component_area:
        return {
            'passed': False,
            'reason': f'no strong handwriting component (max_area={max_area})',
            'ink_pixels': ink_pixels,
            'ink_ratio': round(ink_ratio, 6),
            'max_component_area': max_area,
            'signature_like_components': signature_like_components,
        }

    if signature_like_components == 0:
        return {
            'passed': False,
            'reason': 'no signature-like stroke component',
            'ink_pixels': ink_pixels,
            'ink_ratio': round(ink_ratio, 6),
            'max_component_area': max_area,
            'signature_like_components': signature_like_components,
        }

    return {
        'passed': True,
        'reason': 'signature-like ink present',
        'ink_pixels': ink_pixels,
        'ink_ratio': round(ink_ratio, 6),
        'max_component_area': max_area,
        'signature_like_components': signature_like_components,
    }


def suppress_printed_letters_from_tor_signature(mask: np.ndarray) -> np.ndarray:
    bw = (mask > 0).astype(np.uint8) * 255
    height, width = bw.shape[:2]
    total = float(max(1, height * width))
    bw = remove_tiny_components(bw, min_area=max(3, int(total * 0.00003)))
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, 8)
    seed = np.zeros_like(bw, dtype=np.uint8)
    component_info = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        fill = area / float(max(1, comp_w * comp_h))
        aspect = comp_w / float(max(1, comp_h))
        width_ratio = comp_w / float(max(1, width))
        height_ratio = comp_h / float(max(1, height))
        area_ratio = area / total
        signature_like = (
            (height_ratio >= 0.30 and fill <= 0.55)
            or (width_ratio >= 0.26 and height_ratio >= 0.035 and fill <= 0.45)
            or (aspect >= 3.5 and width_ratio >= 0.18 and fill <= 0.38)
            or (area_ratio >= 0.010 and fill <= 0.35)
        )

        if signature_like:
            seed[labels == label_id] = 255

        component_info.append((
            label_id,
            x,
            y,
            comp_w,
            comp_h,
            area,
            fill,
            aspect,
            width_ratio,
            height_ratio,
            area_ratio,
            signature_like,
        ))

    if np.count_nonzero(seed) < 10 and component_info:
        largest = max(component_info, key=lambda item: item[5])
        seed[labels == largest[0]] = 255

    kernel_width = max(7, int(round(width * 0.055))) | 1
    kernel_height = max(5, int(round(height * 0.075))) | 1
    keep_zone = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height)),
        iterations=1,
    )
    output = np.zeros_like(bw, dtype=np.uint8)

    for component in component_info:
        label_id, _x, _y, _comp_w, _comp_h, _area, fill, aspect, width_ratio, height_ratio, area_ratio, signature_like = component
        comp = labels == label_id
        near_anchor = bool(np.any(keep_zone[comp] > 0))
        blocky_letter = height_ratio <= 0.24 and width_ratio <= 0.24 and fill >= 0.18 and area_ratio <= 0.018
        tiny_letter_piece = height_ratio <= 0.16 and width_ratio <= 0.16 and area_ratio <= 0.010 and fill >= 0.10
        word_fragment = (
            width_ratio >= 0.08
            and width_ratio <= 0.42
            and height_ratio <= 0.20
            and fill >= 0.16
            and aspect >= 1.2
        )
        remove_as_text = (blocky_letter or tiny_letter_piece or word_fragment) and not signature_like

        if remove_as_text:
            if near_anchor and fill < 0.26 and (width_ratio >= 0.18 or height_ratio >= 0.20):
                output[comp] = 255
            continue

        output[comp] = 255

    output = cv2.morphologyEx(
        output,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    final = remove_tiny_components(output, min_area=max(3, int(total * 0.00004)))

    if np.count_nonzero(final) < max(25, int(np.count_nonzero(bw) * 0.18)):
        return bw

    return final


def preprocess_signature_for_model(
    image: np.ndarray,
    is_tor_extracted: bool,
    apply_16x9: bool,
    suppress_printed_letters: bool,
) -> dict:
    binary_mask = to_white_ink_mask(image)
    processed_mask = binary_mask.copy()

    if is_tor_extracted and suppress_printed_letters:
        processed_mask = suppress_printed_letters_from_tor_signature(processed_mask)

    presence = signature_presence_gate(processed_mask)
    model_mask = center_on_16x9_canvas(processed_mask) if apply_16x9 else processed_mask

    return {
        'binary_mask': binary_mask,
        'processed_mask': processed_mask,
        'model_mask': model_mask,
        'presence': presence,
        'preprocess': {
            'is_tor_extracted': is_tor_extracted,
            'binary_white_ink_on_black': True,
            'suppress_printed_letters': bool(is_tor_extracted and suppress_printed_letters),
            'apply_16x9_canvas': bool(apply_16x9),
            'canvas': f'{CANVAS_W}x{CANVAS_H}' if apply_16x9 else None,
            'canvas_fit_scale': CANVAS_FIT_SCALE if apply_16x9 else None,
            'resize': '224x224',
            'normalize': 'mean=0.5, std=0.5',
        },
    }


def center_on_16x9_canvas(
    mask: np.ndarray,
    fit_scale: float = CANVAS_FIT_SCALE,
    pad_ratio: float = CANVAS_PAD_RATIO,
) -> np.ndarray:
    bw = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(bw > 0)
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    if len(xs) == 0 or len(ys) == 0:
        return canvas

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(bw.shape[1], x2 + pad)
    y2 = min(bw.shape[0], y2 + pad)
    crop = bw[y1:y2, x1:x2]
    h, w = crop.shape[:2]

    if h <= 0 or w <= 0:
        return canvas

    max_w = max(1, int(round(CANVAS_W * fit_scale)))
    max_h = max(1, int(round(CANVAS_H * fit_scale)))
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
    if score >= GENUINE_SCORE_THRESHOLD:
        return 'GENUINE'

    if score <= SUSPICIOUS_SCORE_THRESHOLD:
        return 'SUSPICIOUS'

    return 'NEEDS MANUAL REVIEW'


def candidate_presence_failure_to_payload(candidate_presence: dict, decision_threshold: float) -> dict:
    gate_reason = str(candidate_presence.get('reason', 'presence gate failed'))

    return {
        'verdict': 'INVALID',
        'status': 'INVALID',
        'reason': 'no_signature_detected',
        'message': 'No signature detected in the candidate signature region.',
        'signature_detected': False,
        'score': None,
        'distance': None,
        'model_inference_ran': False,
        'presence_failure_detail': gate_reason,
        'candidate_presence': candidate_presence,
        'decision_threshold': round(float(decision_threshold), 4),
        'score_rules': {
            'genuine': f'score >= {GENUINE_SCORE_THRESHOLD:.2f}',
            'suspicious': f'score <= {SUSPICIOUS_SCORE_THRESHOLD:.2f}',
            'needs_manual_review': f'{SUSPICIOUS_SCORE_THRESHOLD:.2f} < score < {GENUINE_SCORE_THRESHOLD:.2f}',
        },
    }


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
