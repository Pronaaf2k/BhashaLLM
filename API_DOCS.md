# BhashaLLM API Documentation

FastAPI backend for Bengali Handwriting Recognition using your trained OCR model.

## Base URL

```
https://your-app-name.onrender.com
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: `https://your-app-name.onrender.com/docs`
- **ReDoc**: `https://your-app-name.onrender.com/redoc`

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
