# Resource Citations & References

This document lists all open-source models, datasets, and research papers used in the **BhashaLLM** project.

## Models

### Large Language Models (LLM)
- **Qwen-2.5-1.5B-Instruct**
  - **Source**: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  - **License**: Apache 2.0
  - **Usage**: Base model for Bangla instruction tuning.
- **XGLM-1.7B** (Evaluated Only)
  - **Source**: [Hugging Face](https://huggingface.co/facebook/xglm-1.7b)
  - **Usage**: Baseline comparison (Discarded due to poor perplexity).

### Optical Character Recognition (OCR)
- **PaddleOCR (PP-OCRv3)**
  - **Source**: [GitHub](https://github.com/PaddlePaddle/PaddleOCR)
  - **License**: Apache 2.0
  - **Usage**: Initial OCR engine choice (Primary Bengali model).
- **Swapnillo/Bangla-OCR-SFT**
  - **Source**: [Hugging Face](https://huggingface.co/swapnillo/Bangla-OCR-SFT)
  - **Architecture**: Qwen3-VL (Vision-Language Model)
  - **Usage**: Evaluated for end-to-end OCR tasks.

## Datasets

### Text Corpora (Phase 1-3)
- **Kazi Nazrul Islam Corpus**
  - **Description**: Poems and literature by Kazi Nazrul Islam.
  - **Source**: Collected from open archives (Project Gutenberg/Wikitext mirrors).
- **Rabindranath Tagore Corpus**
  - **Description**: Selected works of Rabindranath Tagore.
  - **Source**: Open literary archives and public domain texts.

### Instruction Tuning (Phase 5)
- **Synthetic Grading Dataset**
  - **Description**: 200 question-answer pairs generated loosely based on Bangla literature for grading logic training.
  - **Source**: Self-generated (Synthetic).

### OCR Datasets (Phase 4-6)
- **Ekush (Handwritten Characters)**
  - **Source**: [GitHub](https://github.com/ShahariarRabby/ekush) | [Kaggle](https://www.kaggle.com/datasets/shahariar/ekush)
  - **Paper**: ["Ekush: A Multipurpose and Multitype Comprehensive Database for Online Off-Line Bangla Handwritten Characters"](https://link.springer.com/chapter/10.1007/978-981-13-9187-3_14)
  - **Usage**: Planned for fine-tuning OCR models.
- **Bengali.AI Graphemes**
  - **Source**: [Kaggle](https://www.kaggle.com/c/bengaliai-cv19)
  - **Usage**: Reference benchmark dataset (mentioned in ViT paper context).

## Research Papers

1.  **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)**
    - **Authors**: Dosovitskiy et al. (2020)
    - **Link**: [arXiv:2010.01192](https://arxiv.org/abs/2010.01192)
    - **Relevance**: Architecture backbone for modern OCR/VLM systems.

2.  **"LoRA: Low-Rank Adaptation of Large Language Models"**
    - **Authors**: Hu et al. (2021)
    - **Link**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
    - **Relevance**: Method used for efficient fine-tuning of Qwen-2.5.

3.  **"Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities"**
    - **Authors**: Bai et al. (2023)
    - **Link**: [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)
    - **Relevance**: Architecture base for `swapnillo/Bangla-OCR-SFT`.
