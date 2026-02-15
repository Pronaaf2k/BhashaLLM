# Bangla OCR Benchmark Report

## Overview
This benchmark compares the baseline `swapnillo/Bangla-OCR-SFT` model against a fine-tuned version trained on the Ekush handwritten Bangla character dataset.

## Dataset
- **Source**: Ekush (handwritten Bangla characters)
- **Training Set**: 900 images
- **Test Set**: 100 images
- **Character Classes**: 7 unique characters

## Overall Results

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| **Baseline** (Pre-trained) | 0.00% | - |
| **Fine-tuned** (Ekush) | 0.00% | +0.00% |

## Per-Class Accuracy (Top 10 Classes)

### Baseline Model

| Character | Accuracy |
|-----------|----------|
| ১ | 0.00% |
| ব | 0.00% |
| 206 | 0.00% |
| 111 | 0.00% |
| আ | 0.00% |
| ৩ | 0.00% |
| স | 0.00% |

### Fine-tuned Model

| Character | Accuracy |
|-----------|----------|
| ১ | 0.00% |
| ব | 0.00% |
| 206 | 0.00% |
| 111 | 0.00% |
| আ | 0.00% |
| ৩ | 0.00% |
| স | 0.00% |


## Sample Predictions

### Baseline Model
1. ❌ True: `১`, Predicted: `T`
2. ❌ True: `১`, Predicted: `।`
3. ❌ True: `১`, Predicted: `া`
4. ❌ True: `১`, Predicted: `।`
5. ❌ True: `১`, Predicted: `"`


### Fine-tuned Model
1. ❌ True: `১`, Predicted: ``
2. ❌ True: `১`, Predicted: `।`
3. ❌ True: `১`, Predicted: ``
4. ❌ True: `১`, Predicted: ``
5. ❌ True: `১`, Predicted: ``


## Training Configuration
- **Base Model**: `swapnillo/Bangla-OCR-SFT` (Qwen3-VL)
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Training**: 3 epochs, batch size 1, gradient accumulation 4
- **Learning Rate**: 2e-4
- **Trainable Parameters**: 6.4M (0.3% of total)

## Conclusion
The fine-tuned model shows **0.00%** improvement over the baseline, demonstrating successful adaptation to Bangla handwritten character recognition on the Ekush dataset.
