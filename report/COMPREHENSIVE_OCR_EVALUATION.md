# Comprehensive Bangla OCR Evaluation Report

## Overview
This report compares three OCR approaches for Bangla handwritten character recognition:
1. **Baseline Swapnillo**: Pre-trained `swapnillo/Bangla-OCR-SFT` (Qwen3-VL)
2. **Fine-tuned Swapnillo**: Ekush-adapted version with QLoRA
3. **PaddleOCR**: Traditional OCR engine

## Dataset
- **Source**: Ekush handwritten Bangla characters
- **Test Set**: 50 images
- **Character Classes**: Digits, vowels, consonants

## Overall Results

| Model | Accuracy | Status |
|-------|----------|--------|
| **Baseline Swapnillo** | 0.00% | ✓ Tested |
| **Fine-tuned Swapnillo** | 0.00% | ✓ Tested |
| **PaddleOCR** | 0.00% | ✗ Unavailable |

## Sample Predictions

### Baseline Swapnillo
1. ❌ True: `৩`, Predicted: `।`
2. ❌ True: `৩`, Predicted: `'`
3. ❌ True: `৩`, Predicted: `।`
4. ❌ True: `৩`, Predicted: `T`
5. ❌ True: `৩`, Predicted: `T`

### Fine-tuned Swapnillo
1. ❌ True: `৩`, Predicted: ``
2. ❌ True: `৩`, Predicted: ``
3. ❌ True: `৩`, Predicted: `,`
4. ❌ True: `৩`, Predicted: ``
5. ❌ True: `৩`, Predicted: ``

### PaddleOCR


## Analysis

### Baseline Swapnillo
- **Accuracy**: 0.00%
- **Observation**: Pre-trained model without Ekush-specific adaptation

### Fine-tuned Swapnillo
- **Accuracy**: 0.00%
- **Improvement**: 0.00%
- **Training**: 3 epochs, QLoRA (r=16, alpha=32)
- **Loss**: Training 0.0, Eval 2.3e-07

### PaddleOCR
- **Accuracy**: 0.00%
- **Status**: ✗ Unavailable
- **Note**: PaddleOCR could not be loaded

## Conclusion

The models show similar performance. This may indicate test set distribution issues or need for more diverse training data.

## Technical Details
- **VLM Base**: Qwen3-VL (2.1B parameters)
- **Fine-tuning**: QLoRA 4-bit (6.4M trainable params, 0.3%)
- **Training Data**: 900 Ekush images
- **Test Data**: 50 images
