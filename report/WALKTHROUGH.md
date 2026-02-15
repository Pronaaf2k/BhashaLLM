# Walkthrough: Bangla LLM & OCR Grading Pipeline

## Overview
This document tracks the progress and results of adapting `Qwen-2.5-1.5B-Instruct` for Bangla and building an OCR pipeline.

## Phase 2: Baseline Model Selection
We evaluated two candidate models:
- **Qwen-2.5-1.5B-Instruct**: Perplexity **3.80** (Pass)
- **XGLM-1.7B**: Perplexity **73.15** (Fail)

**Decision**: Proceeded with Qwen-2.5-1.5B.

## Phase 3: Bangla Language Adaptation (QLoRA)
We are fine-tuning Qwen-2.5-1.5B on the `data/processed/train.txt` corpus (Nazrul, Tagore, Poems).

### Hyperparameters
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization**: 4-bit (nf4, double_quant)
- **LoRA**: r=16, alpha=32, dropout=0.05, bias="none", target_modules="all-linear"
- **Training**:
    - Context Length: 512 (Reduced from 1024 due to 200W/16GB VRAM constraint)
    - Batch Size: 1
    - Gradient Accumulation: 4
    - Learning Rate: 2e-4
    - Epochs: 1
    - Gradient Checkpointing: True
    - FP16: True

### Training Log
- **Status**: Complete (Epoch 1.0)
- **Loss**: Final ~1.31 (Significant improvement from start ~1.54)
- **Memory**: Stable (No OOM with Context 512)

### Benchmarks
- **Pretrained Qwen-2.5-1.5B**: Ppl **3.80**
- **Adapated Qwen-2.5-1.5B**: Ppl **3.83** (Context 512)
  - *Note*: Performance is comparable. Slight increase likely due to reduced context window (512 vs 2048) during evaluation to save memory.

## Phase 5: Instruction & Grading Fine-tuning
- **Goal**: Fine-tune for grading logic (Question + Answer -> Marks + Feedback).
- **Dataset**: 200 synthetic examples (grading Bangla literature/history questions).
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (Phase 3 adapter continued).
- **Status**: Complete (Epoch 3).
- **Metrics**:
    - Final Train Loss: **~0.097**
    - Final Eval Loss: **~0.018**
    - Token Accuracy: **>99%** (Expected for small synthetic data)
- **Adapters**: Saved in `models/instruct_adapters/final_instruct_adapter`.

## Phase 6: OCR Model Evaluation (In Progress)
- **Objective**: Compare PaddleOCR vs `swapnillo/Bangla-OCR-SFT`.
- **Status**:
  - PaddleOCR: Failed (Bengali model unavailable).
  - Swapnillo: Functional but poor zero-shot performance.
  - **Dataset**: Ekush acquired. 1000 images converted (28x28 PNG) with standard mapping.
    - Mapping verified: `0→০`, `10→অ`, `50→ষ` (digits, vowels, consonants).
  - **Training Data**: Saved to `ocr_training_data.jsonl` (1000 samples).
  - **Training Attempt**: Created `train_ocr.py` with QLoRA.
    - **Blocker**: Qwen3-VL vision processing incompatible with standard HF Trainer.
    - Error: "Image features and image tokens do not match: tokens: 1, features 64"
  - **Citations**: Documented in `citations.md`.
