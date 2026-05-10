# OUR Models

A small Django service for receiving image uploads from a website, preprocessing TOR document images, running forgery inference, and notifying the website when processing is complete.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python manage.py migrate
.\.venv\Scripts\python manage.py runserver
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
      "header": 0.1,
      "body": 0.12,
      "footer": 0.15
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

`patch_baseline_final.pth` is loaded by `images/inference.py` and receives transient patches from `images/preprocessing_pipeline.py`. You can override model settings with:

```powershell
$env:TOR_MODEL_WEIGHTS_PATH = "C:\path\to\patch_baseline_final.pth"
$env:TOR_INFERENCE_THRESHOLD = "0.380"
$env:TOR_INFERENCE_DEVICE = "cpu"
```
