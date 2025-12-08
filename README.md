# BhashaLLM - Bengali Handwriting Recognition API

A FastAPI-based backend API for Bengali handwritten text recognition using your trained OCR model, with Gemini AI integration for tutor and philosophical responses.

## Features

- 📝 **Handwriting Recognition**: Upload Bengali handwriting images via API
- 🔍 **OCR Analysis**: Real-time character recognition using your trained ResNet34 model
- 💬 **AI Chatbot**: Gemini-powered tutor and philosophical responses
- 📊 **Analysis Results**: Detailed metrics and predictions
- 📚 **Literary Perspectives**: Insights from Tagore, Nazrul, and Jasim Uddin
- 📖 **Auto Documentation**: Interactive API docs with Swagger UI and ReDoc

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** For CUDA support on Linux with Python 3.8, install PyTorch separately:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Set Gemini API Key (optional):**
   ```bash
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the API:**
   ```bash
   python app.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:5000/
   ```
   
   Or visit the interactive docs:
   - Swagger UI: http://localhost:5000/docs
   - ReDoc: http://localhost:5000/redoc

### Deploy on Render

1. **Fork/Clone this repository**

2. **Create a new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure the service:**
   - **Build Command:** `pip install -r requirements.txt && pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2`
   - **Environment:** Python 3.8

4. **Add Environment Variables:**
   - Go to "Environment" tab
   - Add `GEMINI_API_KEY` with your API key

5. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment to complete

## Project Structure

```
.
├── app.py                    # FastAPI backend application
├── ocr_model_training.ipynb  # Jupyter notebook for training OCR model
├── requirements.txt          # Python dependencies
├── render.yaml              # Render deployment configuration
├── runtime.txt              # Python version for Render
├── Procfile                 # Alternative deployment file
├── models/                   # Trained OCR models (create this folder)
└── banglaWrittenWordOCR-main/  # Reference implementation (datasets only)
```

## Training the OCR Model

1. Open `ocr_model_training.ipynb` in Jupyter Notebook
2. Run all cells to train the model on `banglaWrittenWordOCR-main` datasets
3. Save the model to `models/ocr_model_final.pth`

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required for chatbot features)

## Notes

- The OCR model uses ResNet34 architecture
- Gemini API handles all text generation (tutor/philosophical responses)
- If model file is not found, the app will use an untrained model (lower accuracy)
- For best results, train the model using the provided notebook

## API Documentation

Once deployed, visit:
- **Swagger UI**: `https://your-app-name.onrender.com/docs`
- **ReDoc**: `https://your-app-name.onrender.com/redoc`

See [API_DOCS.md](API_DOCS.md) for detailed API documentation.

## Troubleshooting

- **Model not found**: Create `models/` folder. The app will work with an untrained model.
- **Gemini API errors**: Make sure `GEMINI_API_KEY` is set correctly
- **Render deployment fails**: Check build logs and ensure all dependencies are in `requirements.txt`
- **Import errors**: Make sure PyTorch is installed (see requirements.txt notes)

## License

MIT License
