# ============================================================
# inference.py
# TOR Document Forgery Detection — Model + Aggregation Only
#
# Preprocessing connects separately later.
#
# Drop this alongside:
#   - patch_baseline_final.pth
#
# Usage:
#   from inference import TORInference
#
#   detector = TORInference(weights_path="patch_baseline_final.pth")
#
#   # patches = list of dicts with keys: "array" (128x128 BGR numpy), "roi" (str)
#   result = detector.predict(patches)
#
#   result.label       -> "genuine" | "fake"
#   result.score       -> float 0-1
#   result.roi_scores  -> {"header": float, "body": float, "footer": float}
#   result.top_roi     -> "body"
#
# Aggregation: flat mean across all patches (ROI scores shown separately for reference)
# Threshold:   0.380
# ============================================================

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image


# ── CONSTANTS ──────────────────────────────────────────────────────────────────
ROIS       = ["header", "body", "footer"]
EMB_DIM    = 1280       # EfficientNet-B0 output dim — do not change
THRESHOLD  = 0.340      # best_threshold from metrics_roi_max.json
IMG_SIZE   = 224
BATCH_SIZE = 32


# ── RESULT ─────────────────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    success:    bool
    label:      Optional[str]    # "genuine" | "fake"
    score:      Optional[float]  # document-level probability
    roi_scores: Optional[dict]   # {"header": float, "body": float, "footer": float}
    top_roi:    Optional[str]
    error:      Optional[str] = None


# ── MODEL ──────────────────────────────────────────────────────────────────────
def _build_model(device):
    backbone = efficientnet_b0(weights=None)

    class PatchBaselineEfficientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone.features
            self.pool     = nn.AdaptiveAvgPool2d(1)
            for p in self.features.parameters():
                p.requires_grad = False
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
            return self.classifier(
                self.pool(self.features(x)).flatten(1)
            ).squeeze(1)

    return PatchBaselineEfficientNet().to(device)


# ── TRANSFORM ──────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── AGGREGATION ────────────────────────────────────────────────────────────────
def _aggregate(patch_probs: list, patch_rois: list) -> dict:
    roi_probs = defaultdict(list)
    for prob, roi in zip(patch_probs, patch_rois):
        roi_probs[roi].append(float(prob))

    roi_scores = {roi: (sum(roi_probs[roi]) / len(roi_probs[roi]) if roi_probs[roi] else 0.0) for roi in ROIS}
    doc_score  = float(sum(patch_probs) / len(patch_probs))   # flat mean across all patches
    top_roi    = max(roi_scores, key=roi_scores.get)

    return {"doc_score": doc_score, "roi_scores": roi_scores, "top_roi": top_roi}


# ── MAIN CLASS ─────────────────────────────────────────────────────────────────
class TORInference:
    """
    Parameters
    ----------
    weights_path : str
        Path to patch_baseline_final.pth
    threshold : float
        Decision threshold. Default is 0.380 (best_threshold from metrics_roi_max.json).
    device : str | None
        "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        weights_path: str,
        threshold:    float = THRESHOLD,
        device:       str   = None,
    ):
        self.device    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.model     = _build_model(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()

    def predict(self, patches: list) -> InferenceResult:
        """
        Parameters
        ----------
        patches : list of dicts
            Each dict must have:
                "array"  -> 128x128 BGR numpy array   (from preprocessor)
                "roi"    -> "header" | "body" | "footer"

        Returns
        -------
        InferenceResult
        """
        if not patches:
            return InferenceResult(
                success=False, label=None, score=None,
                roi_scores=None, top_roi=None,
                error="Empty patch list",
            )

        try:
            probs, rois = self._score_patches(patches)
            agg         = _aggregate(probs, rois)
            label       = "fake" if agg["doc_score"] >= self.threshold else "genuine"

            return InferenceResult(
                success    = True,
                label      = label,
                score      = round(agg["doc_score"], 4),
                roi_scores = {k: round(v, 4) for k, v in agg["roi_scores"].items()},
                top_roi    = agg["top_roi"],
            )
        except Exception as e:
            return InferenceResult(
                success=False, label=None, score=None,
                roi_scores=None, top_roi=None,
                error=str(e),
            )

    def _score_patches(self, patches: list):
        all_probs, all_rois = [], []
        buf_t, buf_r        = [], []

        def _flush():
            if not buf_t:
                return
            batch = torch.stack(buf_t).to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_rois.extend(buf_r)
            buf_t.clear()
            buf_r.clear()

        for p in patches:
            rgb = cv2.cvtColor(p["array"], cv2.COLOR_BGR2RGB)
            t   = _transform(Image.fromarray(rgb))
            buf_t.append(t)
            buf_r.append(p["roi"])
            if len(buf_t) == BATCH_SIZE:
                _flush()

        _flush()
        return all_probs, all_rois
