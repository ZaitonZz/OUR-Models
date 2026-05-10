# OUR Models

A small Django service for receiving image uploads from a website, preprocessing TOR document images, and notifying the website when preprocessing is complete.

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

The upload response includes the Django `job_id`, the website `external_id`, preprocessing status, and the preprocessed image URL.

## Preprocessing callback

After preprocessing finishes, Django POSTs JSON to `callback_url`:

```json
{
  "external_id": "upload-123",
  "job_id": 1,
  "status": "preprocessed",
  "original_image_url": "http://127.0.0.1:8000/media/uploads/...",
  "preprocessed_image_url": "http://127.0.0.1:8000/media/preprocessed/...",
  "method": "brightness",
  "skew_status": "flat",
  "patch_counts": {
    "header": 0,
    "body": 0,
    "footer": 0
  },
  "error": ""
}
```

Check a job later with:

```text
GET /api/images/<id>/
```

## Model hook

`patch_baseline_final.pth` is not run yet. The current flow stops after preprocessing and callback notification. The next phase can feed the transient patches from `images/preprocessing_pipeline.py` into the model.
