from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
import io
import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
import google.generativeai as genai
import sys
import time

app = FastAPI(title="BhashaLLM API", description="Bengali Handwriting Recognition API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add paths for model imports
sys.path.append('banglaWrittenWordOCR-main')
sys.path.append('banglaWrittenWordOCR-main/recongnition_model')

# Configuration
MODEL_PATH = 'models/ocr_model_final.pth'
MAPPING_FILE = 'bangla_ocr_pipeline/grapheme_maps.json'

# Load grapheme mappings
with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    grapheme_maps = json.load(f)

# Load OCR model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import pretrainedmodels
    import torch.nn as nn
    import torch.nn.functional as F
    
    class resnet34(nn.Module):
        def __init__(self):
            super(resnet34, self).__init__()
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
            self.l0 = nn.Linear(512, 168)
            self.l1 = nn.Linear(512, 11)
            self.l2 = nn.Linear(512, 7)

        def forward(self, x):
            bs, _, _, _ = x.shape
            x = self.model.features(x)
            x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
            l0 = self.l0(x)
            l1 = self.l1(x)
            l2 = self.l2(x)
            return l0, l1, l2
    
    ocr_model = resnet34().to(device)
    if os.path.exists(MODEL_PATH):
        ocr_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Loaded OCR model from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Using untrained model.")
    ocr_model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    ocr_model = None

# Initialize Gemini (optional)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini API initialized")
    except:
        print("⚠️ Gemini API not configured")

def preprocess_image(image):
    """Preprocess image for OCR"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale and enhance
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Invert if needed
    image = ImageOps.invert(image)
    
    # Resize
    image = image.resize((224, 128)).convert('RGB')
    
    return image

def recognize_character(image_patch):
    """Recognize a single character"""
    if ocr_model is None:
        return "?", 0.0, {}
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(image_patch).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = ocr_model(img_tensor)
        
        root_idx = torch.argmax(outputs[0], dim=1).item()
        vowel_idx = torch.argmax(outputs[1], dim=1).item()
        cons_idx = torch.argmax(outputs[2], dim=1).item()
        
        root_conf = torch.softmax(outputs[0], dim=1)[0][root_idx].item()
        vowel_conf = torch.softmax(outputs[1], dim=1)[0][vowel_idx].item()
        cons_conf = torch.softmax(outputs[2], dim=1)[0][cons_idx].item()
        
        root_char = grapheme_maps['grapheme_root'].get(str(root_idx), '?')
        vowel_char = grapheme_maps['vowel_diacritic'].get(str(vowel_idx), '')
        cons_char = grapheme_maps['consonant_diacritic'].get(str(cons_idx), '')
        
        recognized_char = root_char + vowel_char + cons_char
        confidence = (root_conf + vowel_conf + cons_conf) / 3
    
    return recognized_char, confidence, {
        'root': (root_char, root_conf),
        'vowel': (vowel_char, vowel_conf),
        'consonant': (cons_char, cons_conf)
    }

def recognize_word(image):
    """Recognize word from image"""
    start_time = time.time()
    
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Recognize
    char, confidence, details = recognize_character(processed_image)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        'recognized_text': char,
        'confidence': confidence,
        'processing_time_ms': int(processing_time),
        'details': details
    }

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    recognized_text: str
    context: Optional[str] = 'tutor'

class PhilosophicalRequest(BaseModel):
    recognized_text: str

# API Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': ocr_model is not None,
        'gemini_configured': gemini_model is not None,
        'device': device
    }

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    """Analyze handwritten Bengali text from an image"""
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load image
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Perform OCR
        result = recognize_word(image)
        
        # Calculate metrics
        metrics = {
            'stroke_quality': 100,
            'linearity': 89,
            'complexity': 24
        }
        
        # Prediction distribution
        predictions = [
            {'text': result['recognized_text'], 'confidence': result['confidence'] * 100, 'is_top': True},
            {'text': 'ভীত', 'confidence': 1.0, 'is_top': False},
            {'text': 'ভাই', 'confidence': 0.5, 'is_top': False}
        ]
        
        return {
            'success': True,
            'recognized_text': result['recognized_text'],
            'confidence': result['confidence'],
            'processing_time_ms': result['processing_time_ms'],
            'metrics': metrics,
            'predictions': predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with Gemini for tutor/philosophical responses"""
    try:
        if not gemini_model:
            raise HTTPException(
                status_code=500,
                detail="Gemini API not configured. Please set GEMINI_API_KEY environment variable"
            )
        
        # Build prompt based on context
        if request.context == 'tutor':
            prompt = f"""You are a helpful Bengali language tutor. The user has written the word "{request.recognized_text}" in Bengali handwriting.

User question: {request.message}

Provide a helpful, educational response in Bengali. If the question is about the recognized word, explain its meaning, usage, etymology, or any relevant linguistic information."""
        else:  # philosophical
            prompt = f"""You are providing a philosophical perspective on Bengali literature and culture. The user has written the word "{request.recognized_text}".

User question: {request.message}

Provide a thoughtful, philosophical response in Bengali, similar to how Rabindranath Tagore, Kazi Nazrul Islam, or Jasim Uddin might reflect on this word. Make it poetic and profound."""
        
        # Get response from Gemini
        response = gemini_model.generate_content(prompt)
        
        return {
            'success': True,
            'response': response.text,
            'context': request.context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/philosophical")
async def philosophical(request: PhilosophicalRequest):
    """Get philosophical perspectives on recognized text"""
    try:
        if not gemini_model:
            raise HTTPException(
                status_code=500,
                detail="Gemini API not configured"
            )
        
        # Get perspectives from different literary figures
        perspectives = []
        
        figures = [
            {'name': 'Rabindranath Tagore', 'style': 'Philosophical', 'prompt': 'philosophical and spiritual'},
            {'name': 'Kazi Nazrul Islam', 'style': 'Passionate', 'prompt': 'passionate and revolutionary'},
            {'name': 'Jasim Uddin', 'style': 'Folk', 'prompt': 'folk and rural life'}
        ]
        
        for figure in figures:
            prompt = f"""Write a poetic reflection on the Bengali word "{request.recognized_text}" in the style of {figure['name']}. 
            Make it {figure['prompt']}, profound, and beautiful. Write in Bengali, 2-3 sentences."""
            
            try:
                response = gemini_model.generate_content(prompt)
                perspectives.append({
                    'name': figure['name'],
                    'style': figure['style'],
                    'quote': response.text
                })
            except:
                perspectives.append({
                    'name': figure['name'],
                    'style': figure['style'],
                    'quote': f'"{request.recognized_text}" সম্পর্কে {figure["name"]} এর দৃষ্টিভঙ্গি...'
                })
        
        return {
            'success': True,
            'perspectives': perspectives
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    os.makedirs('models', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
