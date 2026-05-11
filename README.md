# OUR Models

A small Django service for receiving image uploads from a website, preprocessing TOR document images, running forgery inference, and notifying the website when processing is complete.

## Setup

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python manage.py migrate
.\.venv\Scripts\python manage.py runserver
```

Linux server:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install --no-cache-dir -r requirements.txt
python manage.py migrate
export DJANGO_ALLOWED_HOSTS="127.0.0.1,localhost,172.18.0.1"
python manage.py runserver 127.0.0.1:8001
```

Open `http://127.0.0.1:8000/` for the upload form.

## Upload API

Send `multipart/form-data` to `POST /api/images/` with:

- `image`: uploaded image file.
- `external_id`: unique id owned by the website.
- `callback_url`: website endpoint Django should notify after preprocessing.

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" -F "external_id=upload-123" -F "callback_url=https://example.com/api/results" http://127.0.0.1:8000/api/images/
```

The upload response includes the Django `job_id`, the website `external_id`, processing status, the preprocessed image URL, and the inference result.

## Processing callback

After preprocessing and inference finish, Django POSTs JSON to `callback_url`:

```json
{
  "external_id": "upload-123",
  "job_id": 1,
  "status": "complete",
  "original_image_url": "http://127.0.0.1:8000/media/uploads/...",
  "preprocessed_image_url": "http://127.0.0.1:8000/media/preprocessed/...",
  "method": "brightness",
  "skew_status": "flat",
  "patch_counts": {
    "header": 0,
    "body": 0,
    "footer": 0
  },
  "result": {
    "success": true,
    "label": "genuine",
    "score": 0.1234,
    "roi_scores": {
      "header": {"n_patches": 1, "top5_mean": 0.1},
      "body": {"n_patches": 2, "top5_mean": 0.12},
      "footer": {"n_patches": 3, "top5_mean": 0.15}
    },
    "top_roi": "footer",
    "error": ""
  },
  "error": ""
}
```

Check a job later with:

```text
GET /api/images/<id>/
```

## Model

`model1_fulltrain_final.pth` is loaded by `images/inference_efficientnet_topk.py` and receives transient patches from `images/preprocessing_pipeline.py`. It scores the whole document by averaging the top 5 patch probabilities across header, body, and footer. You can override model settings with:

```powershell
$env:TOR_EFFICIENTNET_TOPK_MODEL_WEIGHTS_PATH = "C:\path\to\model1_fulltrain_final.pth"
$env:TOR_EFFICIENTNET_TOPK_INFERENCE_THRESHOLD = "0.800"
$env:TOR_EFFICIENTNET_TOPK_INFERENCE_DEVICE = "cpu"
$env:TOR_EFFICIENTNET_TOPK_TOP_K = "5"
```
