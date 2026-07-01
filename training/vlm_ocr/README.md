# VLM OCR Training Toolkit

This folder contains scripts to fine-tune a local VLM for Bangla handwritten OCR.

Chosen base model:

```text
google/paligemma2-3b-pt-448
```

Why this model:

- Current low-tier Google open VLM closest to the Gemini/Gemma family for local fine-tuning.
- 3B parameter size is practical for local LoRA/QLoRA experiments.
- PaliGemma 2 is explicitly intended for text reading/OCR-style fine-tuning tasks.
- Unlike Gemini API models, it has downloadable trainable weights.

## Folder

```text
training/vlm_ocr/
├── config.json
├── requirements-vlm.txt
├── prepare_dataset.py
├── check_dataset.py
├── train_paligemma_lora.py
├── infer_paligemma.py
├── evaluate_paligemma.py
└── data/
```

## Data Format

Start with a CSV:

```csv
image,text
images/line_001.jpg,দেশের সাহসী ছেলেরা স্বাধীনতার জন্য জীবন
images/line_002.jpg,উৎসর্গ করেন। তারা মুক্তিযুদ্ধের শহীদ।
```

Best training unit: **line-level images**.

Do not start with full pages unless you already have exact page-level transcriptions. Line-level OCR is easier and usually gives faster improvement.

## Install

PaliGemma 2 is gated on Hugging Face. Before training, accept Google's Gemma license on the model page and run:

```bash
huggingface-cli login
```

From the `BhashaLLM` root:

```bash
cd /home/benaaf/Desktop/BhashaLLM
source .venv/bin/activate
python -m pip install -r training/vlm_ocr/requirements-vlm.txt
```

## Prepare Dataset

Flexible importer for most local datasets:

```bash
python training/vlm_ocr/import_dataset.py \
  --input /path/to/your/dataset \
  --out-dir training/vlm_ocr/data
```

Supported inputs:

- CSV with `image,text`, `image_path,text`, or `path,label`
- JSONL with `image/text`, `image_path/text`, or `path/label`
- Folder of images where each image has a matching `.txt` file with the transcription

Examples:

```bash
python training/vlm_ocr/import_dataset.py --input /data/bangla_lines/labels.csv
python training/vlm_ocr/import_dataset.py --input /data/bangla_lines/labels.jsonl
python training/vlm_ocr/import_dataset.py --input /data/bangla_lines/images_with_txt_labels
```

Older CSV-specific script:

```bash
python training/vlm_ocr/prepare_dataset.py \
  --csv /path/to/line_labels.csv \
  --image-root /path/to/image/root \
  --out-dir training/vlm_ocr/data
```

This creates:

```text
training/vlm_ocr/data/train.jsonl
training/vlm_ocr/data/val.jsonl
training/vlm_ocr/data/test.jsonl
```

Validate:

```bash
python training/vlm_ocr/check_dataset.py training/vlm_ocr/data/train.jsonl
python training/vlm_ocr/check_dataset.py training/vlm_ocr/data/val.jsonl
python training/vlm_ocr/check_dataset.py training/vlm_ocr/data/test.jsonl
```

## Train LoRA/QLoRA

Default uses 4-bit loading.

```bash
python training/vlm_ocr/train_paligemma_lora.py \
  --config training/vlm_ocr/config.json
```

Outputs:

```text
training/vlm_ocr/outputs/paligemma2_3b_448_bangla_ocr_lora/
```

Final adapter:

```text
training/vlm_ocr/outputs/paligemma2_3b_448_bangla_ocr_lora/final_adapter
```

## Inference

Base model only:

```bash
python training/vlm_ocr/infer_paligemma.py \
  --image /path/to/line_or_page.jpg
```

Fine-tuned adapter:

```bash
python training/vlm_ocr/infer_paligemma.py \
  --image /path/to/line_or_page.jpg \
  --adapter training/vlm_ocr/outputs/paligemma2_3b_448_bangla_ocr_lora/final_adapter
```

## Evaluate

```bash
python training/vlm_ocr/evaluate_paligemma.py \
  --jsonl training/vlm_ocr/data/test.jsonl \
  --adapter training/vlm_ocr/outputs/paligemma2_3b_448_bangla_ocr_lora/final_adapter \
  --out training/vlm_ocr/outputs/test_predictions.jsonl
```

Metrics:

- `cer`: character error rate
- `wer`: word error rate

Lower is better.

## Training Advice

- Start with 500-1000 labeled line images to verify the pipeline.
- Move to 10k+ line images for meaningful improvement.
- Keep labels exact. Do not train correction and OCR at the same time.
- Use held-out pages/writers for test data.
- If the model repeats text, reduce `max_new_tokens` in `config.json`.
- If GPU memory fails, reduce `image_max_pixels`, keep `batch_size=1`, and increase gradient accumulation.

## Next Step

Create a labeled CSV of cropped handwritten Bangla lines, then run `prepare_dataset.py`.
