# OCR Training Expansion Implementation Plan

## Objective
The goal is to drastically improve the performance of our Qwen2.5-VL-based Bangla OCR model by incorporating all relevant handwriting datasets listed in our `llmdocreadme.md` resource list. 

## Datasets to Incorporate
Based on the "Awesome Bangla Datasets" repository, we will add/re-enable the following sets alongside our baseline `banglawriting` and `ekush`:

1.  **Bongabdo (Handwritten English & Bengali)**
    *   **Source:** HuggingFace `deepcopy/handwritten-text-recognition-bongabdo`
    *   **Current Status:** Downloaded by `prepare_ocr_datasets.py`, but explicitly commented out in `train_ocr_improved.py` due to "tokenization issues with full sentences".
    *   **Action Plan:** Re-enable this dataset and update the dataset collator/tokenization logic inside `bhasha.data.dataset.OCRDataset` to robustly handle full sentence lengths.

2.  **BN-HTRd (Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition)**
    *   **Source:** Mendeley Data (`743k6dm543/4`)
    *   **Rationale:** Perfect for our use case! It focuses on *document-level* handwritten text, which closely mirrors reading full student answers.
    *   **Action Plan:** We will write a script to download, extract, and convert the BN-HTRd dataset format into our standard `train.jsonl` format containing image paths and text.

3.  **PDF Text Detection Dataset**
    *   **Source:** Kaggle `warcoder/bangla-text-detection-and-recognition`
    *   **Rationale:** Can help the model learn to read clean typed texts if we ever need it to process typed worksheets. 
    *   **Action Plan:** If Kaggle API is available, we will pull this. However, since student grading is our primary focus, we will prioritize BN-HTRd.

## Implementation Steps

### Step 1: Automate Dataset Downloads & Formatting
*   Modify `prepare_ocr_datasets.py` (or create a new `prepare_advanced_ocr_data.py`) to systematically fetch the **BN-HTRd** dataset (and any Kaggle sets if API credentials exist).
*   Parse the bounding boxes/transcriptions into our standardized `.jsonl` structure (`{"image": "...", "text": "..."}`).

### Step 2: Fix Max Length & Tokenization for Sentences
*   Review and refactor `bhasha/data/dataset.py` (specifically `OCRDataset` and `collate_fn`). 
*   We need to ensure long sequences (like Bongabdo or BN-HTRd sentences) do not get truncated aggressively and are padded correctly using the `AutoProcessor`.

### Step 3: Update Training Script
*   Update `train_ocr_improved.py` to concatenate all generated `.jsonl` datasets (Ekush, Banglawriting, Bongabdo, BN-HTRd).
*   Run the training loop in the `venv` to generate our final super-charged student-grading OCR adapter.

## Final Review
Once the OCR model can transcribe full, messy handwritten sentences properly, the student grading pipeline will be much more accurate.
