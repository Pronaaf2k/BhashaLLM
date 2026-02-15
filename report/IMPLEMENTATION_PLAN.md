# Implementation Plan: Bangla LLM & OCR Grading Pipeline

## Goal
Design, train, and evaluate a small (~1.5B parameter) Bangla language model and an OCR-assisted grading pipeline within `/home/node/.openclaw/workspace/BhashaLLM`.

## Proposed Changes

### Phase 1: Data Preparation
#### [NEW] [data_prep.py](file:///home/node/.openclaw/workspace/BhashaLLM/data_prep.py)
- **Inputs**: `data/raw/nazrul.csv`, `data/raw/tagore.csv`, `data/raw/poems.csv`.
- **Normalization**:
    - Unicode normalization (NFKC).
    - Bangla specific punctuation (e.g., `|` -> `।`).
- **Splitting**: 80/10/10 into `data/processed/{train,val,test}.txt`.

### Phase 3: Bangla Language Adaptation (QLoRA) [Completed]
- **Status**: Done. Loss ~1.31. Adapters saved.

### Phase 4: OCR Pipeline
#### [MODIFY] `ocr_pipeline.py`
- Added PaddleOCR and Tesseract wrappers.
- **Issue**: Tesseract missing, PaddleOCR Bengali model download failed.
- **Correction**: Pivoted to `swapnillo/Bangla-OCR-SFT` (Qwen-VL).

### Phase 5: Instruction & Grading Fine-tuning [Completed]
- **Status**: Done. Validation loss ~0.018.

### Phase 6: OCR Model Evaluation & Training (Current)
- **Goal**: Evaluate and Fine-tune `swapnillo/Bangla-OCR-SFT`.
- **Dataset**: "Ekush" (CSVs converted to images).
- **Mapping**: Verifying label ID mapping using VLM inference.

### Phase 6 & 7: Evaluation
#### [NEW] [evaluate.py](file:///home/node/.openclaw/workspace/BhashaLLM/evaluate.py)
- Metrics: Perplexity, BLEU/ROUGE, CER/WER, Pearson Correlation (grading).

## Verification Plan
### Automated Tests
- `python data_prep.py` : [x] Verified (6.6M tokens).
- `python train_clm.py --dry_run` : Ensure model fits in VRAM (16GB).
- `python ocr_pipeline.py --image <image>` : Verify OCR output.
- `python train_instruct.py --epochs 0.1` : Smoke test instruction tuning.

### Manual Verification
- Inspect 5 random samples from filtered dataset.
- Review 5 generated grading feedbacks.
