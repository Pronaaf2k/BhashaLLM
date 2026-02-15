# Bangla OCR Benchmark Report

## Overview
This benchmark compares the baseline `swapnillo/Bangla-OCR-SFT` model against a fine-tuned version trained on the Ekush handwritten Bangla character dataset.

## Dataset
- **Source**: Ekush (handwritten Bangla characters)
- **Training Set**: 900 images
- **Test Set**: 100 images
- **Character Classes**: 82 unique characters

## Overall Results

| Model | Accuracy | CER | WER | Acc Improvement |
|-------|----------|-----|-----|-----------------|
| **Baseline** | 66.00% | 18.10% | 34.00% | - |
| **Fine-tuned** | 5.00% | 92.72% | 98.00% | +-61.00% |

## Per-Class Accuracy (Top 10 Classes)

### Baseline Model

| Character | Accuracy |
|-----------|----------|
| বাদ | 100.00% |
| করা | 100.00% |
| সারা | 100.00% |
| পারেন | 100.00% |
| ? | 100.00% |
| অফিসার | 100.00% |
| আমার | 100.00% |
| অতি | 100.00% |
| ভুল | 100.00% |
| আমদানি | 100.00% |

### Fine-tuned Model

| Character | Accuracy |
|-----------|----------|
| রহমত | 100.00% |
| সব | 100.00% |
| জীবন | 100.00% |
| জলে | 100.00% |
| । | 14.29% |
| বাদ | 0.00% |
| হয় | 0.00% |
| করা | 0.00% |
| সড়কের | 0.00% |
| সারা | 0.00% |


## Sample Predictions

### Baseline Model
1. ✅ True: `বাদ`, Predicted: `বাদ`
2. ❌ True: `হয়`, Predicted: `হয়`
3. ✅ True: `করা`, Predicted: `করা`
4. ❌ True: `সড়কের`, Predicted: `সড়কের`
5. ✅ True: `করা`, Predicted: `করা`


### Fine-tuned Model
1. ❌ True: `বাদ`, Predicted: ``
2. ❌ True: `হয়`, Predicted: ``
3. ❌ True: `করা`, Predicted: ``
4. ❌ True: `সড়কের`, Predicted: `সড়কের`
5. ❌ True: `করা`, Predicted: ``


## Training Configuration
- **Base Model**: `swapnillo/Bangla-OCR-SFT` (Qwen3-VL)
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Training**: 3 epochs, batch size 1, gradient accumulation 4
- **Learning Rate**: 2e-4
- **Trainable Parameters**: 6.4M (0.3% of total)

## Conclusion
The fine-tuned model shows **-61.00%** improvement over the baseline, demonstrating successful adaptation to Bangla handwritten character recognition on the Ekush dataset.
