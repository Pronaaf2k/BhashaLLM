# BhashaLLM API Documentation

The server exposes **two** surfaces. Everything below the "Legacy
endpoints" heading is the original application; the `/api/v1` surface is
the pipeline the paper describes.

| Prefix | What it is | In the paper? |
| --- | --- | --- |
| `/api/v1/*` | Base model + one resident QLoRA adapter. Local, no network calls. | Yes — Sec. III-F |
| `/ui` | Optional single-page frontend, opt-in via `BHASHA_ENABLE_UI=1`. | Yes — Sec. III-F |
| `/api/analyze`, `/api/chat`, `/api/philosophical` | ResNet-34 grapheme classifier + Gemini cloud calls. | No — see `ERRATA.md` C3 |

> **On the base URL below.** The original document documents a
> `your-app-name.onrender.com` deployment. Render is a cloud host, and
> Sections III-F and VI-F build an argument on the system running locally
> and offline — "without depending on cloud APIs or continuous network
> access". Together with the Gemini calls (C3) and the retrieval stack
> (C5), this is the third artifact in the repository that assumes network
> access. The local base URL is `http://localhost:5000`. Recorded in
> `ERRATA.md` C26.

## Base URL

```
http://localhost:5000              # python main.py
https://your-app-name.onrender.com # if deployed to a cloud host
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation at `/docs` (Swagger
UI) and `/redoc`.

---

## `/api/v1` — the paper's pipeline

All local. Decoding defaults are Section IV-C's fixed settings
(temperature 0.7, top-p 0.9, repetition penalty 1.1, 256 new tokens for
text; greedy with 128 tokens for OCR).

### `GET /api/v1/status`

Resident models, active adapter, VRAM, and which adapters are present on
disk. `vram_allocated_gb` should not grow across adapter swaps — that is
the evidence for the residency claim in Sec. III-F.

### `POST /api/v1/generate`

```json
{"prompt": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে লিখুন।",
 "adapter": "bangla", "max_new_tokens": 256, "return_logprobs": false}
```

`adapter` is `bangla`, `grading`, `ocr`, or `null` for the bare base
model. `return_logprobs` emits the per-token log-probabilities that
`eval/confidence.py` turns into the Table VII confidence score.

### `POST /api/v1/grade`

```json
{"question": "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
 "reference_answer": "তিনি একজন বাঙালি কবি ও সাহিত্যিক।",
 "student_answer": "তিনি একজন লেখক।"}
```

Loads the grading adapter and applies ChatML automatically. The prompt is
Bangla-only, per Sec. IV-C.

### `POST /api/v1/ocr`

Multipart upload, field `image`. Query parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `image_size` | 256 | Square edge. Matches training preprocessing. |
| `return_confidence` | true | Computes `eval/confidence.py`'s definition. |
| `use_adapter` | true | `false` gives Table VII's **"Before"** baseline. |

### `POST /api/v1/adapter`

```json
{"adapter": "grading"}
```

Swaps the resident adapter explicitly and returns the VRAM before and
after. `null` unloads everything.

---

## Legacy endpoints

These are the original application and are **not** part of the paper's
methodology. `/api/chat` and `/api/philosophical` call the Gemini cloud
API and require `GEMINI_API_KEY`.

## Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gemini_configured": true,
  "device": "cpu"
}
```

### 2. Analyze Handwriting

**POST** `/api/analyze`

Analyze Bengali handwritten text from an image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `image`: Image file (PNG, JPG, JPEG)

**Example (cURL):**
```bash
curl -X POST https://your-app-name.onrender.com/api/analyze \
  -F "image=@handwriting.jpg"
```

**Response:**
```json
{
  "success": true,
  "recognized_text": "ভাত",
  "confidence": 0.98,
  "processing_time_ms": 150,
  "metrics": {
    "stroke_quality": 100,
    "linearity": 89,
    "complexity": 24
  },
  "predictions": [
    {
      "text": "ভাত",
      "confidence": 98.0,
      "is_top": true
    },
    {
      "text": "ভীত",
      "confidence": 1.0,
      "is_top": false
    },
    {
      "text": "ভাই",
      "confidence": 0.5,
      "is_top": false
    }
  ]
}
```

### 3. Chat with Gemini (Tutor/Philosophical)

**POST** `/api/chat`

Get AI responses about the recognized text.

**Request:**
```json
{
  "message": "এই শব্দটির অর্থ কী?",
  "recognized_text": "ভাত",
  "context": "tutor"  // or "philosophical"
}
```

**Response:**
```json
{
  "success": true,
  "response": "ভাত হল বাংলার প্রধান খাদ্য...",
  "context": "tutor"
}
```

### 4. Get Philosophical Perspectives

**POST** `/api/philosophical`

Get literary perspectives from Bengali poets.

**Request:**
```json
{
  "recognized_text": "ভাত"
}
```

**Response:**
```json
{
  "success": true,
  "perspectives": [
    {
      "name": "Rabindranath Tagore",
      "style": "Philosophical",
      "quote": "ভাত শুধু অন্ন নয়..."
    },
    {
      "name": "Kazi Nazrul Islam",
      "style": "Passionate",
      "quote": "ভাত! এ শুধু ক্ষুধা নিবারণ নয়..."
    },
    {
      "name": "Jasim Uddin",
      "style": "Folk",
      "quote": "আহা, ভাত! মায়ের হাতের..."
    }
  ]
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

Status codes:
- `400`: Bad Request (missing parameters)
- `500`: Internal Server Error

## Example Usage (Python)

```python
import requests

# Analyze image
with open('handwriting.jpg', 'rb') as f:
    response = requests.post(
        'https://your-app-name.onrender.com/api/analyze',
        files={'image': f}
    )
    result = response.json()
    print(f"Recognized: {result['recognized_text']}")

# Chat about recognized text
chat_response = requests.post(
    'https://your-app-name.onrender.com/api/chat',
    json={
        'message': 'এই শব্দটির ব্যুৎপত্তি কী?',
        'recognized_text': result['recognized_text'],
        'context': 'tutor'
    }
)
print(chat_response.json()['response'])
```

## Example Usage (JavaScript/Fetch)

```javascript
// Analyze image
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('https://your-app-name.onrender.com/api/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Recognized:', result.recognized_text);
```

## Example Usage (Python with httpx)

```python
import httpx

# Analyze image
with open('handwriting.jpg', 'rb') as f:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://your-app-name.onrender.com/api/analyze',
            files={'image': f}
        )
        result = response.json()
        print(f"Recognized: {result['recognized_text']}")
```

## Environment Variables

- `GEMINI_API_KEY`: Required for chat and philosophical endpoints
- `PORT`: Automatically set by Render

## Notes

- The model loads from `models/ocr_model_final.pth`
- If model not found, uses untrained model (lower accuracy)
- All image processing is done server-side
- Responses are in JSON format
- FastAPI automatically generates OpenAPI schema at `/openapi.json`
