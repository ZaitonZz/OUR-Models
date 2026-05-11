from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


ROIS = ['header', 'body', 'footer']
EMB_DIM = 1280
IMG_SIZE = 224
BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.800
DEFAULT_TOP_K = 5
DEFAULT_AGGREGATION = 'topk_mean'


@dataclass
class InferenceResult:
    success: bool
    label: Optional[str]
    score: Optional[float]
    roi_scores: Optional[dict]
    top_roi: Optional[str]
    top_roi_score: Optional[float]
    aggregation: Optional[str]
    threshold: Optional[float]
    error: Optional[str] = None


def _build_model(device):
    backbone = efficientnet_b0(weights=None)

    class PatchBaselineEfficientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)

            for parameter in self.features.parameters():
                parameter.requires_grad = False

            self.classifier = nn.Sequential(
                nn.Linear(EMB_DIM, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            x = self.classifier(x)
            return x.squeeze(1)

    return PatchBaselineEfficientNet().to(device)


def _safe_load_state_dict(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _mean_top_k(values: list[float], top_k: int) -> float:
    if not values:
        return 0.0

    scores = np.array(values, dtype=np.float32)
    k = min(top_k, len(scores))

    return float(np.mean(np.sort(scores)[-k:]))


def _aggregate(
    patch_probs: list[float],
    patch_rois: list[str],
    aggregation: str = DEFAULT_AGGREGATION,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if len(patch_probs) == 0:
        return {
            'doc_score': 0.0,
            'roi_scores': {},
            'top_roi': None,
            'top_roi_score': 0.0,
        }

    all_scores = [float(probability) for probability in patch_probs]
    roi_probs = defaultdict(list)

    for probability, roi in zip(patch_probs, patch_rois):
        roi = str(roi).lower().strip()

        if roi not in ROIS:
            roi = 'unknown'

        roi_probs[roi].append(float(probability))

    if aggregation == 'mean':
        doc_score = float(np.mean(all_scores))
    elif aggregation == 'max':
        doc_score = float(np.max(all_scores))
    elif aggregation == 'topk_mean':
        doc_score = _mean_top_k(all_scores, top_k)
    elif aggregation == 'roi_max':
        roi_max_scores = [
            float(np.max(roi_probs[roi]))
            for roi in ROIS
            if roi_probs[roi]
        ]
        doc_score = float(max(roi_max_scores)) if roi_max_scores else 0.0
    elif aggregation == 'roi_topk_mean':
        roi_topk_scores = [
            _mean_top_k(roi_probs[roi], top_k)
            for roi in ROIS
            if roi_probs[roi]
        ]
        doc_score = float(max(roi_topk_scores)) if roi_topk_scores else 0.0
    else:
        raise ValueError(
            f'Unknown aggregation: {aggregation}. '
            'Use mean, max, topk_mean, roi_max, or roi_topk_mean.'
        )

    roi_scores = {}

    for roi in ROIS:
        scores = roi_probs[roi]

        if not scores:
            roi_scores[roi] = {
                'n_patches': 0,
                'mean': 0.0,
                'max': 0.0,
                f'top{top_k}_mean': 0.0,
            }
            continue

        roi_scores[roi] = {
            'n_patches': int(len(scores)),
            'mean': float(np.mean(scores)),
            'max': float(np.max(scores)),
            f'top{top_k}_mean': _mean_top_k(scores, top_k),
        }

    top_roi = max(ROIS, key=lambda region: roi_scores[region][f'top{top_k}_mean'])
    top_roi_score = float(roi_scores[top_roi][f'top{top_k}_mean'])

    return {
        'doc_score': float(doc_score),
        'roi_scores': roi_scores,
        'top_roi': top_roi,
        'top_roi_score': top_roi_score,
    }


class TORInference:
    def __init__(
        self,
        weights_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        aggregation: str = DEFAULT_AGGREGATION,
        top_k: int = DEFAULT_TOP_K,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = float(threshold)
        self.aggregation = aggregation
        self.top_k = int(top_k)
        self.model = _build_model(self.device)

        state_dict = _safe_load_state_dict(weights_path, self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, patches: list) -> InferenceResult:
        if not patches:
            return InferenceResult(
                success=False,
                label=None,
                score=None,
                roi_scores=None,
                top_roi=None,
                top_roi_score=None,
                aggregation=self.aggregation,
                threshold=self.threshold,
                error='Empty patch list.',
            )

        try:
            patch_probs, patch_rois = self._score_patches(patches)
            aggregate = _aggregate(
                patch_probs=patch_probs,
                patch_rois=patch_rois,
                aggregation=self.aggregation,
                top_k=self.top_k,
            )

            doc_score = float(aggregate['doc_score'])
            label = 'fake' if doc_score >= self.threshold else 'genuine'
            rounded_roi_scores = {}

            for roi, scores in aggregate['roi_scores'].items():
                rounded_roi_scores[roi] = {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in scores.items()
                }

            return InferenceResult(
                success=True,
                label=label,
                score=round(doc_score, 4),
                roi_scores=rounded_roi_scores,
                top_roi=aggregate['top_roi'],
                top_roi_score=round(float(aggregate['top_roi_score']), 4),
                aggregation=self.aggregation,
                threshold=self.threshold,
                error=None,
            )
        except Exception as exc:
            return InferenceResult(
                success=False,
                label=None,
                score=None,
                roi_scores=None,
                top_roi=None,
                top_roi_score=None,
                aggregation=self.aggregation,
                threshold=self.threshold,
                error=str(exc),
            )

    def _score_patches(self, patches: list):
        all_probs, all_rois = [], []
        batch_tensors, batch_rois = [], []

        def flush_batch():
            if not batch_tensors:
                return

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch)).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_rois.extend(batch_rois)
            batch_tensors.clear()
            batch_rois.clear()

        for patch in patches:
            if 'array' not in patch:
                raise ValueError("Each patch must contain an 'array' key.")

            if 'roi' not in patch:
                raise ValueError("Each patch must contain a 'roi' key.")

            roi = str(patch['roi']).lower().strip()

            if roi not in ROIS:
                raise ValueError(f"Invalid ROI '{roi}'. Expected one of: {ROIS}")

            rgb = cv2.cvtColor(patch['array'], cv2.COLOR_BGR2RGB)
            tensor = _transform(Image.fromarray(rgb))

            batch_tensors.append(tensor)
            batch_rois.append(roi)

            if len(batch_tensors) >= BATCH_SIZE:
                flush_batch()

        flush_batch()

        return all_probs, all_rois
