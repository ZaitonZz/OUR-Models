# OUR Models

A small Django service for receiving image uploads from a website and processing them with model code.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python manage.py migrate
.\.venv\Scripts\python manage.py runserver
```

Open `http://127.0.0.1:8000/` for the upload form.

## Upload API

Send `multipart/form-data` to `POST /api/images/` with the file field named `image`.

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" http://127.0.0.1:8000/api/images/
```

To have this app POST the processed result to another API, include `callback_url`:

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" -F "callback_url=https://example.com/api/results" http://127.0.0.1:8000/api/images/
```

Or set a default callback endpoint before running the server:

```powershell
$env:RESULTS_API_URL = "https://example.com/api/results"
.\.venv\Scripts\python manage.py runserver
```

The callback JSON looks like:

```json
{
  "id": 1,
  "status": "complete",
  "image_url": "http://127.0.0.1:8000/media/uploads/...",
  "result": {},
  "error": ""
}
```

Check a job later with:

```text
GET /api/images/<id>/
```

## Model hook

Put your model pipeline in `images/services.py` inside `run_models(image_path)`. Return a JSON-serializable dictionary so Django can save it and return it from the API.
