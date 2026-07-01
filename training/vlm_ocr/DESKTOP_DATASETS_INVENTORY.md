# Desktop Datasets Inventory

Source root:

```text
/home/benaaf/Desktop/datasets
```

Purpose of this note: record what is inside the Desktop `datasets` folder and identify which data is suitable for training the existing PaliGemma Bangla OCR adapter.

## Summary Table

| Dataset folder | Images | Metadata | Main use | PaliGemma line OCR suitability |
|---|---:|---|---|---|
| `image_787 to image_927` | 1,401 | `combined_bhashallm_dataset.csv` | Line-level Bangla OCR image/text pairs | High, but 52 CSV image references are missing |
| `bnaf` | 224 | `bnaf.csv` | Line/sentence-level Bangla OCR image/text pairs | High, all 224 CSV pairs valid |
| `dataset10thMay-20260514T174723Z-3-001` | 157 | `dataset10thMay/dataset10thmay.csv` | Line-level Bangla OCR image/text pairs | High, all 152 CSV pairs valid |
| `bengali-handwritten-text-with-bounding-boxes` | 260 | `metadata.csv`, `annotations_json/*.json` | Full/page images with word/region boxes and labels | Medium/high after converting boxes to line/word crops |
| `bangla-handwritten-datatset` | 547,941 | `dataset/labels/label.csv` | Word-level handwritten Bangla recognition | Medium; useful for word OCR or mixed fine-tuning, not line-first |
| `banglalekha` | 166,105 | `metadata.jsonl`, Readme | Character-class handwritten Bangla | Low for line OCR; useful for character classifier/pretraining only |
| `bangla-handwriting-dataset-for-pix2pix` | 17,946 | manifests and schema JSON | Image-to-image handwriting restoration pairs | Low for direct OCR; useful for preprocessing/restoration experiments |
| `Dataset HAI` | 400 | none found | Line-looking images | Not directly trainable until labels/transcriptions are added |

## Direct PaliGemma Training Candidates

### `dataset10thMay-20260514T174723Z-3-001`

Path:

```text
/home/benaaf/Desktop/datasets/dataset10thMay-20260514T174723Z-3-001/dataset10thMay
```

Metadata:

```text
dataset10thmay.csv
columns: filename,text
rows: 152
valid image/text pairs: 152
missing images: 0
empty text: 0
bad images: 0
```

Image properties sampled:

```text
extension: .png
typical size: 2480 x 400
```

Example rows:

```text
img1_line1.png -> কাদামাটির রাস্তা
img1_line2.png -> বৃষ্টির পরে গ্রামের রাস্তা কাদায় ভরে
img1_line3.png -> গেল। স্কুলে যাওয়ার সময় তিথি
```

Use this as clean line-level OCR data. Good for validation/smoke testing because all CSV pairs resolve.

### `bnaf`

Path:

```text
/home/benaaf/Desktop/datasets/bnaf
```

Metadata:

```text
bnaf.csv
columns: filename,text
rows: 224
valid image/text pairs: 224
missing images: 0
empty text: 0
bad/corrupt images: 0
unreferenced images: 0
```

Dataset type:

```text
line/sentence-level OCR crops
```

Image properties:

```text
extension: .png
width: 1044 for all validated rows
height: 400 for all validated rows
```

Text statistics:

```text
character length min/median/mean/max: 4 / 26 / 24.61 / 35
word count min/median/mean/max: 1 / 5 / 4.50 / 7
```

Example rows:

```text
img1line1.png -> সততার মূল্য
img1line2.png -> সততা মানুষের সবচেয়ে বড় গুণগুলোর
img1line3.png -> একটি। মিথ্যা বলে সাময়িক লাভ
img1line4.png -> হওয়া
img1line5.png -> যায়, একজন সৎ মানুষ ভুল করলে
```

Use this in the first PaliGemma line OCR training mix. It is smaller than `image_787 to image_927` but fully valid and cleaner than datasets with missing references.

### `image_787 to image_927`

Path:

```text
/home/benaaf/Desktop/datasets/image_787 to image_927
```

Metadata:

```text
combined_bhashallm_dataset.csv
columns: serial,File name,extracted text
rows: 1377
valid image/text pairs: 1325
missing images: 52
empty text: 0
bad images: 0
```

Image properties sampled:

```text
extension scan found: .png files only
CSV references include some .jpg filenames that are missing
sampled width range: 218 to 2112
sampled median width: 1750
sampled height range: 105 to 449
sampled median height: 173
```

Example rows:

```text
img787line1.png -> পুঁজিবাজারে তালিকাভুক্ত সিমেন্ট খাতের কোম্পানি
img787line2.png -> লাফার্জ সুরমা সিমেন্ট ১:১ রাইট শেয়ার
img787line4.png -> কমিশন (এসইসি) আজ বৃহস্পতিবার কমিশনের
```

Use this as the larger line-level OCR training set after skipping/fixing the 52 missing image references.

## Other Useful Datasets

### `bengali-handwritten-text-with-bounding-boxes`

Path:

```text
/home/benaaf/Desktop/datasets/bengali-handwritten-text-with-bounding-boxes
```

Metadata:

```text
metadata.csv
columns: filename,image_path,annotation_path,image_width,image_height,image_mode,num_annotations,unique_labels_count,labels,total_annotation_area,annotation_density
rows: 260
annotation files: annotations_json/*.json
images: 260 .jpg
```

This is a detection + recognition dataset. It is not immediately in image/text-line JSONL format, but it can be converted into crops using the JSON annotations. It may be valuable for word/line crop generation and OCR detection benchmarking.

### `bangla-handwritten-datatset`

Path:

```text
/home/benaaf/Desktop/datasets/bangla-handwritten-datatset
```

Metadata:

```text
dataset/labels/label.csv
columns: image_id,text
rows: 547941
valid image/text pairs: 547940
empty text: 1
```

Image properties sampled:

```text
extension: .jpg
sampled width range: 11 to 446
sampled median width: 147
sampled height range: 14 to 168
sampled median height: 67
```

This is a large word-level dataset. It can help PaliGemma learn Bangla word recognition, but it should not be the first dataset if the target is full line/page transcription.

### `banglalekha`

Path:

```text
/home/benaaf/Desktop/datasets/banglalekha
```

Metadata:

```text
metadata.jsonl
rows: 166105
fields include: label,district,institution_id,gender,age,month_day,form_serial,filename
```

This is a handwritten character dataset with 84 classes. It is useful for character recognition or classifier pretraining, not direct PaliGemma line OCR.

### `bangla-handwriting-dataset-for-pix2pix`

Path:

```text
/home/benaaf/Desktop/datasets/bangla-handwriting-dataset-for-pix2pix
```

Metadata:

```text
total samples: 17946
train: 14356
val: 1794
test: 1796
format: pix2pix AB PNG, 384 x 256
```

Task: image-to-image handwriting restoration, not OCR transcription. Useful for preprocessing/restoration experiments before OCR, not directly for PaliGemma text generation unless labels are added.

## Unlabeled Line-Looking Image Folders

### `Dataset HAI`

```text
images: 400 .png
metadata: none found
sampled size: 2480 x 400
```

This looks like line crops but has no transcription file. It cannot be used for supervised PaliGemma OCR until labels are created or recovered.

## Recommended PaliGemma Plan

1. Start with the fully valid line-level datasets: `bnaf` (224/224 valid pairs) and `dataset10thMay` (152/152 valid pairs).
2. Add valid rows from `image_787 to image_927` after skipping/fixing 52 missing image references; this gives 1325 valid extra line samples.
3. Train the existing `google/paligemma2-3b-pt-448` LoRA path into a new output directory such as `training/vlm_ocr/outputs/paligemma2_3b_448_desktop_lines_lora`.
4. Evaluate on the held-out test split and record CER/WER.
5. Keep `bangla-handwritten-datatset` as optional later mixed word-level data.
6. Convert `bengali-handwritten-text-with-bounding-boxes` annotations into crops if more supervised OCR data is needed.
7. Do not use `Dataset HAI` for supervised training until labels are available.

## Graphs To Track In Notebook

Dataset graphs:

- image count by dataset folder
- selected line samples by source
- train/val/test split sizes
- text character length distribution
- text word count distribution
- image width/height scatter
- missing image count by source

Training graphs:

- train loss by step
- eval loss by step
- learning rate by step if available

Evaluation graphs:

- CER distribution
- WER distribution
- source-wise CER/WER after predictions are tagged by dataset source
- worst examples table by CER
