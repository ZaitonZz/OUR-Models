# ============================================================
# preprocessing_pipeline.py
# TOR Document Preprocessing - App / Inference Only
#
# Drop this + tor_preprocessor_config.json into your app.
#
# Usage:
#   from preprocessing_pipeline import DocumentPreprocessor
#
#   pipeline = DocumentPreprocessor.load_config("tor_preprocessor_config.json")
#   result   = pipeline.run("uploaded_photo.jpg")
#
#   result.warped        -> standardized 1024x1700 image (numpy array)
#   result.patches       -> list of 128x128 patch arrays + spatial coords
#   result.patch_counts  -> {"header": N, "body": N, "footer": N}
#   result.method        -> which crop method fired
#   result.skew_status   -> "flat" | "deskewed"
#
# Pipeline order (genuine):
#   1. Detect & crop document (perspective transform)
#   2. Deskew if needed (>= min_skew_angle degrees)
#   3. CLAHE contrast adjustment
#   4. Gaussian + Median denoising
#   5. Standardize to 1024x1700
#   6. Patch extraction
# ============================================================

import cv2
import numpy as np
import uuid
import os
import json
from dataclasses import dataclass, field
from typing import Optional


# -- CONFIG -------------------------------------------------------------------

@dataclass
class PipelineConfig:
    ratio_min:      float = 0.50
    ratio_max:      float = 0.75
    min_area_frac:  float = 0.10
    max_area_frac:  float = 0.90
    target_w:       int   = 1024
    target_h:       int   = 1700
    patch_size:     int   = 128
    stride:         int   = 64
    # CLAHE
    clahe_clip:     float = 2.0
    clahe_tile:     int   = 8       # used as (clahe_tile, clahe_tile)
    # Denoising
    gaussian_ksize: int   = 3       # used as (gaussian_ksize, gaussian_ksize)
    gaussian_sigma: float = 0.8
    median_ksize:   int   = 3
    denoise_alpha:  float = 0.5     # blend: alpha*gaussian + (1-alpha)*median
    # Deskew
    min_skew_angle: float = 3.0     # degrees; below this -> skip warp
    rois: dict = field(default_factory=lambda: {
        "header": {"x1": 10,  "y1": 85,   "x2": 1015, "y2": 495 },
        "body":   {"x1": 10,  "y1": 460,  "x2": 1015, "y2": 1420},
        "footer": {"x1": 10,  "y1": 1300, "x2": 1015, "y2": 1620},
    })


# -- RESULT -------------------------------------------------------------------

@dataclass
class PreprocessResult:
    success:      bool
    method:       str                  # 'brightness' | 'brightness+hull' | 'edges' | 'no_crop' | 'failed'
    skew_status:  str                  # 'flat' | 'deskewed'
    warped:       Optional[np.ndarray] # standardized 1024x1700 image
    patches:      list                 # list of patch dicts (see below)
    patch_counts: dict                 # {"header": N, "body": N, "footer": N}
    error:        Optional[str] = None

# Each patch dict:
#   array        -> 128x128 BGR numpy array   (feed this to the model)
#   roi          -> 'header' | 'body' | 'footer'
#   doc_x/doc_y  -> top-left position in the full 1024x1700 document
#   patch_row    -> row index in the ROI grid
#   patch_col    -> col index in the ROI grid
#   roi_x1/y1    -> ROI top-left in the document
#   roi_x2/y2    -> ROI bottom-right in the document
#   patch_size   -> 128
#   stride       -> 64


# -- MAIN CLASS ---------------------------------------------------------------

class DocumentPreprocessor:

    def __init__(self, config: PipelineConfig = None):
        self.cfg = config or PipelineConfig()

    @classmethod
    def load_config(cls, path: str):
        """Load from tor_preprocessor_config.json and return ready pipeline."""
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        return cls(config=PipelineConfig(**cfg_dict))

    # -- RUN ------------------------------------------------------------------

    def run(
        self,
        image_path: str = None,
        image_array: np.ndarray = None,
    ) -> PreprocessResult:
        """
        Preprocess a single uploaded TOR image.

        Pass either image_path (file path) or image_array (numpy BGR array).

        Pipeline:
          1. Detect & crop document (perspective transform)
          2. Deskew if skew >= min_skew_angle
          3. CLAHE contrast adjustment
          4. Gaussian + Median denoising
          5. Standardize to target_w x target_h
          6. Extract patches

        Returns PreprocessResult with warped image and patches ready for model.
        """
        image = self._load(image_path, image_array)
        if image is None:
            return PreprocessResult(
                success=False, method="failed", skew_status="n/a",
                warped=None, patches=[], patch_counts={},
                error=f"Could not read image: {image_path}"
            )

        # Step 1 - Crop & perspective correct
        cropped, method = self._crop(image)

        # Steps 2-4 - Genuine enhancement (deskew -> CLAHE -> denoise)
        enhanced, skew_status = self._enhance(cropped)

        # Step 5 - Standardize to 1024x1700
        standardized = self._standardize(enhanced)

        # Step 6 - Patch extraction
        patches, counts = self._extract_patches(standardized, image_path)

        return PreprocessResult(
            success=True,
            method=method,
            skew_status=skew_status,
            warped=standardized,
            patches=patches,
            patch_counts=counts,
        )

    # -- CROP PREVIEW ---------------------------------------------------------

    def crop_preview(
        self,
        image_path: str = None,
        image_array: np.ndarray = None,
    ):
        """
        Crop only - use for the UI preview step.
        Returns (cropped_image, method).
        """
        image = self._load(image_path, image_array)
        if image is None:
            return None, "failed"
        return self._crop(image)

    def crop_manual(
        self,
        corners: np.ndarray,
        image_path: str = None,
        image_array: np.ndarray = None,
    ):
        """
        If user adjusted corners in the UI, call this instead of crop_preview.
        corners: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=float32)
        Returns (cropped_image, "manual").
        """
        image = self._load(image_path, image_array)
        if image is None:
            return None, "failed"
        image  = self._rotate_to_portrait(image)
        warped = self._four_point_transform(image, corners)
        warped = self._rotate_to_portrait(warped)
        return warped, "manual"

    # -- INTERNALS ------------------------------------------------------------

    def _load(self, path, array):
        if array is not None:
            return array.copy()
        if path:
            return cv2.imread(path)
        return None

    def _crop(self, image):
        image = self._rotate_to_portrait(image)

        pts, method = self._find_document(image)

        if pts is None:
            return image, "no_crop"

        warped = self._four_point_transform(image, pts)
        warped = self._rotate_to_portrait(warped)

        return warped, method

    def _enhance(self, image):
        """
        Genuine preprocessing steps - applied after crop, before standardize.

        Step 2 - Deskew (Hough line detection, only if skew >= min_skew_angle)
        Step 3 - CLAHE on L channel in LAB space
        Step 4 - Blended Gaussian + Median denoising
        """
        # Step 2 - Deskew
        skew = self._measure_skew(image)
        if abs(skew) >= self.cfg.min_skew_angle:
            h, w   = image.shape[:2]
            center = (w // 2, h // 2)
            M      = cv2.getRotationMatrix2D(center, skew, 1.0)
            cos_v  = abs(M[0, 0])
            sin_v  = abs(M[0, 1])
            new_w  = int(h * sin_v + w * cos_v)
            new_h  = int(h * cos_v + w * sin_v)
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            image  = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            skew_status = "deskewed"
        else:
            skew_status = "flat"

        # Step 3 - CLAHE (L channel in LAB space)
        lab        = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
        )
        l_enhanced = clahe.apply(l)
        image      = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        # Step 4 - Blended Gaussian + Median denoising
        ksize  = (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize)
        gauss  = cv2.GaussianBlur(image, ksize, self.cfg.gaussian_sigma)
        median = cv2.medianBlur(image, self.cfg.median_ksize)
        alpha  = self.cfg.denoise_alpha
        image  = cv2.addWeighted(gauss, alpha, median, 1.0 - alpha, 0)

        image = np.clip(image, 0, 255).astype(np.uint8)
        return image, skew_status

    def _standardize(self, image):
        """Step 5 - Resize and center onto a clean 1024x1700 white canvas."""
        h, w    = image.shape[:2]
        scale   = min(self.cfg.target_w / w, self.cfg.target_h / h)
        new_w   = max(1, int(w * scale))
        new_h   = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas  = np.full((self.cfg.target_h, self.cfg.target_w, 3), 255, dtype=np.uint8)
        x_off   = (self.cfg.target_w - new_w) // 2
        y_off   = (self.cfg.target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def _extract_patches(self, image, image_path):
        """Step 6 - Extract overlapping 128x128 patches per ROI."""
        patches = []
        counts  = {}
        h, w    = image.shape[:2]
        img_id  = (
            os.path.splitext(os.path.basename(image_path))[0]
            if image_path else uuid.uuid4().hex[:8]
        )

        for roi_name, roi in self.cfg.rois.items():
            x1 = max(0, min(roi["x1"], w))
            x2 = max(0, min(roi["x2"], w))
            y1 = max(0, min(roi["y1"], h))
            y2 = max(0, min(roi["y2"], h))
            roi_img = image[y1:y2, x1:x2]

            if roi_img.shape[0] < self.cfg.patch_size or roi_img.shape[1] < self.cfg.patch_size:
                counts[roi_name] = 0
                continue

            count = 0
            for row_idx, py in enumerate(
                range(0, roi_img.shape[0] - self.cfg.patch_size + 1, self.cfg.stride)
            ):
                for col_idx, px in enumerate(
                    range(0, roi_img.shape[1] - self.cfg.patch_size + 1, self.cfg.stride)
                ):
                    patch = roi_img[py:py + self.cfg.patch_size, px:px + self.cfg.patch_size]
                    patches.append({
                        "array":      patch,
                        "roi":        roi_name,
                        "doc_x":      x1 + px,
                        "doc_y":      y1 + py,
                        "patch_row":  row_idx,
                        "patch_col":  col_idx,
                        "roi_x1":     x1,
                        "roi_y1":     y1,
                        "roi_x2":     x2,
                        "roi_y2":     y2,
                        "patch_size": self.cfg.patch_size,
                        "stride":     self.cfg.stride,
                    })
                    count += 1
            counts[roi_name] = count

        return patches, counts

    def _measure_skew(self, image):
        """
        Estimates document skew angle via Hough line detection on text lines.
        Returns angle in degrees. Already-flat images return ~0.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10,
        )
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    @staticmethod
    def _rotate_to_portrait(image):
        h, w = image.shape[:2]
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image

    def _find_document(self, image):
        pts, method = self._detect_via_brightness(image)
        if pts is not None:
            return pts, method
        pts, method = self._detect_via_edges(image)
        if pts is not None:
            return pts, method
        return None, "failed"

    def _detect_via_brightness(self, image):
        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        cnts, _   = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts      = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        image_area = image.shape[0] * image.shape[1]
        for c in cnts:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and self._is_valid_ratio(approx.reshape(4, 2), image_area):
                return approx.reshape(4, 2).astype("float32"), "brightness"
            hull   = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4 and self._is_valid_ratio(approx.reshape(4, 2), image_area):
                return approx.reshape(4, 2).astype("float32"), "brightness+hull"
        return None, None

    def _detect_via_edges(self, image):
        ratio      = image.shape[0] / 800.0
        small      = cv2.resize(image, (int(image.shape[1] / ratio), 800))
        gray       = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred    = cv2.GaussianBlur(gray, (7, 7), 0)
        small_area = small.shape[0] * small.shape[1]
        for lo, hi in [(30, 100), (50, 150), (75, 200), (10, 50)]:
            edged   = cv2.Canny(blurred, lo, hi)
            edged   = cv2.dilate(edged, None, iterations=1)
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            for c in cnts:
                peri   = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and self._is_valid_ratio(approx.reshape(4, 2), small_area):
                    return (approx.reshape(4, 2) * ratio).astype("float32"), "edges"
        return None, None

    def _is_valid_ratio(self, pts, image_area):
        rect   = self._order_points(pts.astype("float32"))
        tl, tr, br, bl = rect
        width  = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br))
        if height == 0 or width == 0:
            return False
        area_frac = cv2.contourArea(pts) / image_area
        ratio     = width / height
        return (
            self.cfg.ratio_min <= ratio <= self.cfg.ratio_max and
            self.cfg.min_area_frac <= area_frac <= self.cfg.max_area_frac
        )

    @staticmethod
    def _order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _four_point_transform(image, pts):
        rect = DocumentPreprocessor._order_points(pts)
        tl, tr, br, bl = rect
        maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst  = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
        M    = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxW, maxH))
