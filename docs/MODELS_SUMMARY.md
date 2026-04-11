# BhashaLLM Models - Complete Setup Summary

## ✅ All Models Downloaded and Ready!

### Base Models (Downloaded from HuggingFace)
These are now cached locally in `~/.cache/huggingface/hub/` and won't need to be re-downloaded:

1. **Qwen/Qwen2.5-1.5B-Instruct** (2.9GB)
   - Used for: Bangla LLM and Instruction/Grading models
   - Location: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct`

2. **swapnillo/Bangla-OCR-SFT** (4.0GB)
   - Used for: OCR model
   - Location: `~/.cache/huggingface/hub/models--swapnillo--Bangla-OCR-SFT`

### Your Fine-Tuned Adapters (Local)
These are your trained models in `models/`:

1. **Bangla Language Adapter** (1.5GB)
   - Path: `models/bangla_adapters/final_adapter/`
   - Purpose: Bangla literature/artist trained LLM
   - Checkpoints: 10, 100, 200, 300, 400, 417

2. **Instruction/Grading Adapter** (1.4GB)
   - Path: `models/instruct_adapters/final_instruct_adapter/`
   - Purpose: Grading Bangla answers
   - Checkpoints: 20, 40, 60, 80, 100, 120, 135

3. **OCR Adapter** (743MB)
   - Path: `models/ocr_adapters/banglawriting_adapter/`
   - Purpose: Handwritten Bangla character recognition
   - Checkpoints: 50, 600, 675

## 🚀 How to Use Your Models

### 1. Test All Models
```bash
python3 test_models.py
```
This will verify all three models are working correctly.

### 2. Test OCR with Adapter
```bash
python3 bhasha/scripts/evaluate_swapnillo.py --adapter_path models/ocr_adapters/banglawriting_adapter
```

### 3. Test Bangla LLM
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Load your adapter
model = PeftModel.from_pretrained(model, "models/bangla_adapters/final_adapter")

# Generate text
prompt = "আমি বাংলায় গান গাই"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 4. Test Grading Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/instruct_adapters/final_instruct_adapter")

# Load your adapter
model = PeftModel.from_pretrained(model, "models/instruct_adapters/final_instruct_adapter")

# Grade an answer
prompt = "Grade this answer: বাংলাদেশের রাজধানী ঢাকা।"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📊 Storage Summary

| Component | Size | Location |
|-----------|------|----------|
| Base Models (cached) | 6.9GB | `~/.cache/huggingface/hub/` |
| Your Adapters | 3.6GB | `models/` |
| **Total** | **10.5GB** | |

## 🔧 Maintenance

### Re-download Base Models
If you ever need to re-download the base models:
```bash
python3 download_base_models.py
```

### Clear HuggingFace Cache
To free up space (will require re-downloading):
```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct
rm -rf ~/.cache/huggingface/hub/models--swapnillo--Bangla-OCR-SFT
```

## ✅ Verification Status

All models have been tested and verified working:
- ✅ Bangla LLM generates Bangla text
- ✅ Grading model provides feedback in Bangla
- ✅ OCR model base + adapter loaded successfully

## 📝 Notes

- No HuggingFace token needed for future runs (models are cached)
- All paths use relative references (portable across systems)
- Models are excluded from Git (in `.gitignore`)
- Use virtual environment: `source venv/bin/activate`

## 🎯 Next Steps

1. **Retrain OCR** with fixed scripts:
   ```bash
   python3 bhasha/ocr/train.py
   ```

2. **Evaluate Models**:
   ```bash
   python3 bhasha/eval/evaluate.py
   python3 bhasha/eval/all_ocr.py
   ```

3. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Fixed OCR pipeline and updated paths"
   git push origin main
   ```
