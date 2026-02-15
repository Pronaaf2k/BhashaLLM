# Task: Bangla LLM & OCR Grading Pipeline

> [!IMPORTANT]
> **RESTART CHECKPOINT:**
> - **Last Action**: Paused XGLM evaluation due to hardware heat.
> - **Next Step**: Retry XGLM evaluation with strict resource limits.
> - **Status**: All scripts (`*.py`) and processed data (`data/processed/`) are saved on disk.
> - **Constraint**: GPU Power Usage limited < 200W (Batch Size 1, Context 1024).

- [x] **Phase 1: Data Understanding & Preparation**
    - [x] dataset_setup: Acquire Nazrul, Tagore, and Stylometric datasets
    - [x] standardization: Normalize text
    - [x] splitting: Create Train/Val/Test splits
    - [x] stats: Compute token counts (~6.6M), vocab coverage (~26k)
- [ ] **Phase 2: Baseline Model Selection**
    - [x] selection: Selected Qwen-2.5-1.5B-Instruct & XGLM-1.7B
    - [x] baseline_eval: Qwen-2.5-1.5B Ppl=3.80. XGLM-1.7B Ppl=73.15
- [ ] **Phase 3: Bangla Language Adaptation (QLoRA)**
    - [x] training_setup: Created train_clm.py (Optimized for <200W)
    - [x] pretraining: Run continued pretraining on Bangla corpus (Loss ~1.31)
    - [x] saving: Save adapters and logs
    - [ ] **Hyperparameters**:
        - Model: `Qwen/Qwen2.5-1.5B-Instruct` (4-bit nf4, double_quant)
        - LoRA: r=16, alpha=32, dropout=0.05, bias="none", all-linear
        - Training: Batch=1, GradAcc=4, LR=2e-4, Context=512, FP16, GradCheckpoint=True
- [ ] **Phase 4: OCR Pipeline**
    - [x] ocr_engine: Installed PaddleOCR (Working). Tesseract (System missing).
    - [x] preprocessing: Implemented in ocr_pipeline.py
    - [x] ensemble: Implemented in ocr_pipeline.py
    - [x] correction: Stub implemented
    - [ ] **Research**: Evaluate `swapnillo/Bangla-OCR-SFT` vs PaddleOCR (User Request)
    - [x] **Dataset**: Ekush acquired & processed (1000 images, standard mapping).
    - [/] **Training**: Fine-tune `swapnillo/Bangla-OCR-SFT` (74% complete, loss 0.0).
    - [ ] **Evaluation**: Compare baseline vs fine-tuned on test set.
    - [ ] **Benchmark**: Generate comprehensive report with metrics.
- [ ] **Phase 5: Instruction & Grading Fine-tuning**
    - [x] synthetic_data: Generated 200 sample grading pairs
    - [x] instruct_tuning: Fine-tune on synthetic grading data (Complete)
- [ ] **Phase 6: Evaluation**
    - [x] eval_lm: Qwen-Adapter Ppl=3.83 (Context 512). Baseline=3.80.
    - [ ] eval_text: BLEU/ROUGE scores
    - [ ] eval_vector: ViT (arXiv:2010.00170) dataset is "Bengali.AI Graphemes" (Kaggle).
    - [ ] eval_ocr: CER/WER metrics
    - [ ] eval_grading: Correlation with human rubrics
- [ ] **Phase 7: Analysis & Reporting**
    - [ ] reporting: Generate comparison tables & plots
    - [x] documentation: Final summary & future work
    - [x] citations: Created citations.md (Models, Datasets, Papers)
