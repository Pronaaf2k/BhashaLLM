# OCR Dataset Report

Source folder:

```text
/home/benaaf/Desktop/ocr_datasets
```

The datasets are already unified into:

```text
/home/benaaf/Desktop/ocr_datasets/ocr_unified_manifest.csv
/home/benaaf/Desktop/ocr_datasets/splits/manifest_train.csv
/home/benaaf/Desktop/ocr_datasets/splits/manifest_val.csv
/home/benaaf/Desktop/ocr_datasets/splits/manifest_test.csv
```

## What Exists

- `bangla-handwritten-datatset`: word-level handwritten Bangla recognition.
- `csai-hcr`: line/paragraph-level handwritten Bangla recognition.
- `bengali-handwritten-text-with-bounding-boxes`: small page/bbox detection + recognition set.
- `banglalekha`: character-class images, useful for classifier/pretraining but not direct text transcription.
- `bangla-handwriting-dataset-for-pix2pix`: image-to-image cleanup/preprocessing, not direct OCR labels.

## Split Counts

Train split:

- `recognition_word`: 493,022 rows
- `recognition_line_or_paragraph`: 895 rows
- `detection_and_recognition`: 231 rows
- `recognition_character_class`: 149,340 rows
- `image_to_image_pix2pix`: 16,152 rows

Validation/test splits also exist.

## Best Use For VLM OCR

Best direct VLM training data:

- `recognition_line_or_paragraph` from `csai-hcr`

Why:

- It is image-to-text at line/paragraph level.
- This matches the target behavior of a VLM: image in, Bangla text out.

Limitation:

- Only 895 train samples, 54 val, 47 test.
- Good for proof-of-concept, not enough for a strong production handwritten OCR VLM.

Created VLM-ready JSONL:

```text
training/vlm_ocr/data_csai_lines/train.jsonl  # 889 rows
training/vlm_ocr/data_csai_lines/val.jsonl    # 54 rows
training/vlm_ocr/data_csai_lines/test.jsonl   # 47 rows
```

## Best Use For Word OCR

Best word-level data:

- `recognition_word` from `bangla-handwritten-datatset`

Why:

- Huge volume: ~493k train word images.
- Good for a recognizer model or VLM word-reading adapter.

Limitation:

- Word-level training alone does not solve full-page OCR.
- You still need detection/line segmentation or a page-level model.

Created sampled VLM-ready JSONL:

```text
training/vlm_ocr/data_words_sample/train.jsonl  # 50,000 rows
training/vlm_ocr/data_words_sample/val.jsonl    # 5,000 rows
training/vlm_ocr/data_words_sample/test.jsonl   # 5,000 rows
```

## Recommended Training Strategy

1. Smoke-test VLM fine-tuning on `data_csai_lines`.
2. If the training loop works, mix in word data from `data_words_sample`.
3. For real page OCR, collect/crop more handwritten line images from your target notebook style.
4. Keep a fixed held-out test set from your real notebook photos.

## Commands Used

Line/paragraph dataset:

```bash
python training/vlm_ocr/import_ocr_datasets.py \
  --task-types recognition_line_or_paragraph \
  --out-dir training/vlm_ocr/data_csai_lines
```

Word sample dataset:

```bash
python training/vlm_ocr/import_ocr_datasets.py \
  --task-types recognition_word \
  --out-dir training/vlm_ocr/data_words_sample \
  --limit-train 50000 \
  --limit-val 5000 \
  --limit-test 5000 \
  --max-text-len 80
```

Validation:

```bash
python training/vlm_ocr/check_dataset.py training/vlm_ocr/data_csai_lines/train.jsonl
python training/vlm_ocr/check_dataset.py training/vlm_ocr/data_words_sample/train.jsonl --limit 1000
```
