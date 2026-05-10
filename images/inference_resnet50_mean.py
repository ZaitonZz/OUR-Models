from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


ROIS = ['header', 'body', 'footer']
EMB_DIM = 2048
THRESHOLD = 0.340
IMG_SIZE = 224
BATCH_SIZE = 32


@dataclass
class InferenceResult:
    success: bool
    label: Optional[str]
    score: Optional[float]
    roi_scores: Optional[dict]
    top_roi: Optional[str]
    error: Optional[str] = None


def _build_model(device):
    backbone = resnet50(weights=None)

    class PatchBaselineResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*list(backbone.children())[:-2])
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

    return PatchBaselineResNet50().to(device)


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _aggregate(patch_probs: list, patch_rois: list) -> dict:
    roi_probs = defaultdict(list)

    for prob, roi in zip(patch_probs, patch_rois):
        roi_probs[roi].append(float(prob))

    roi_scores = {
        roi: sum(roi_probs[roi]) / len(roi_probs[roi]) if roi_probs[roi] else 0.0
        for roi in ROIS
    }
    doc_score = float(sum(patch_probs) / len(patch_probs))
    top_roi = max(roi_scores, key=roi_scores.get)

    return {
        'doc_score': doc_score,
        'roi_scores': roi_scores,
        'top_roi': top_roi,
    }


class TORInference:
    def __init__(
        self,
        weights_path: str,
        threshold: float = THRESHOLD,
        device: str = None,
    ):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.threshold = threshold
        self.model = _build_model(self.device)

        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)

        self.model.eval()

    def predict(self, patches: list) -> InferenceResult:
        if not patches:
            return InferenceResult(
                success=False,
                label=None,
                score=None,
                roi_scores=None,
                top_roi=None,
                error='Empty patch list',
            )

        try:
            probs, rois = self._score_patches(patches)
            agg = _aggregate(probs, rois)
            label = 'fake' if agg['doc_score'] >= self.threshold else 'genuine'

            return InferenceResult(
                success=True,
                label=label,
                score=round(agg['doc_score'], 4),
                roi_scores={key: round(value, 4) for key, value in agg['roi_scores'].items()},
                top_roi=agg['top_roi'],
            )
        except Exception as exc:
            return InferenceResult(
                success=False,
                label=None,
                score=None,
                roi_scores=None,
                top_roi=None,
                error=str(exc),
            )

    def _score_patches(self, patches: list):
        all_probs, all_rois = [], []
        buf_t, buf_r = [], []

        def flush():
            if not buf_t:
                return

            batch = torch.stack(buf_t).to(self.device)

            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch)).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_rois.extend(buf_r)
            buf_t.clear()
            buf_r.clear()

        for patch in patches:
            rgb = cv2.cvtColor(patch['array'], cv2.COLOR_BGR2RGB)
            tensor = _transform(Image.fromarray(rgb))

            buf_t.append(tensor)
            buf_r.append(patch['roi'])

            if len(buf_t) == BATCH_SIZE:
                flush()

        flush()
        return all_probs, all_rois
