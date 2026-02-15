# BhashaLLM: Bangla OCR & Language Model Pipeline

This project implements a pipeline for Bangla handwritten OCR using Qwen-VL (fine-tuned) and a Bangla language model.

## 📂 Project Structure

- `data/`: Contains processed datasets (`train.jsonl`, `val.jsonl`, `test.jsonl`).
- `models/`: Stores model adapters and checkpoints.
- `report/`: Evaluation reports and benchmarks.
- `*.py`: Training and evaluation scripts.

## 🚀 Setup

### 1. Environment
Recommended to use Conda or Python venv (Python 3.12).

**Option A: Conda**
```bash
conda env create -f environment.yml
conda activate bhashallm
```

**Option B: pip**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dependencies
Ensure you have `paddlepaddle-gpu` installed if you want to use PaddleOCR for comparison.
```bash
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

## 🏃 Usage

### 1. Data Preparation
If starting from raw data, run:
```bash
python convert_banglawriting.py
```
This extracts images from the raw dataset and prepares `data/processed/banglawriting/`.

### 2. OCR Training
To fine-tune the OCR model (Qwen-VL adapter):
```bash
python train_ocr.py --max_steps 300 --batch_size 1 --grad_accum 16
```
- `max_steps`: Total training steps.
- `batch_size`: Per-device batch size (1 recommended for variable image sizes).

### 3. Evaluation
To benchmark the model (metrics: CER, WER, Accuracy):
```bash
python evaluate_ocr_models.py
```
This will compare the baseline `swapnillo/Bangla-OCR-SFT` with your trained adapter and generate `report/ocr_benchmark_report.md`.

## 📊 Results
Check `report/antigravity_experiments/` for logs and generated artifacts.

## 📄 References
- **Aye T. Maung et al.**: Hybrid YOLO+CNN approach.
- **Apurba**: Grapheme decomposition and CER/WER metrics.
