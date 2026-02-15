#!/usr/bin/env python3
"""
Comprehensive OCR Evaluation: Baseline vs Fine-tuned vs PaddleOCR
"""
import json
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
from ekush_mapping import get_label_text
import os
from tqdm import tqdm

def load_baseline_model(model_id):
    """Load baseline Swapnillo model."""
    print(f"Loading baseline model: {model_id}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def load_finetuned_model(model_id, adapter_path):
    """Load fine-tuned Swapnillo model."""
    print(f"Loading fine-tuned model from: {adapter_path}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def load_paddleocr():
    """Load PaddleOCR model."""
    try:
        from paddleocr import PaddleOCR
        print("Loading PaddleOCR (attempting Bengali, fallback to English)")
        # Try Bengali first
        try:
            ocr = PaddleOCR(lang='bn', use_angle_cls=True, show_log=False)
            print("✓ PaddleOCR loaded with Bengali model")
            return ocr, 'bn'
        except:
            # Fallback to English
            ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)
            print("⚠ PaddleOCR loaded with English model (Bengali unavailable)")
            return ocr, 'en'
    except Exception as e:
        print(f"✗ PaddleOCR failed to load: {e}")
        return None, None

def predict_swapnillo(model, processor, image_path):
    """Predict using Swapnillo model."""
    image = Image.open(image_path).convert('RGB')
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What character is this?"}
        ]
    }]
    
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    generated_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    answer = generated_text.split("assistant")[-1].strip()
    return answer

def predict_paddleocr(ocr, image_path):
    """Predict using PaddleOCR."""
    if ocr is None:
        return "[PaddleOCR unavailable]"
    
    try:
        result = ocr.ocr(image_path, cls=True)
        if result and result[0]:
            # Extract text from first detection
            text = result[0][0][1][0]
            return text
        return "[No text detected]"
    except Exception as e:
        return f"[Error: {str(e)[:20]}]"

def evaluate_model(model_type, model, processor_or_ocr, test_data, max_samples=50):
    """Evaluate a model on test data."""
    correct = 0
    total = 0
    samples = []
    predictions = []
    
    for item in tqdm(test_data[:max_samples], desc=f"Evaluating {model_type}"):
        image_path = item['image']
        true_label = item['text']
        
        try:
            if model_type == "PaddleOCR":
                pred_label = predict_paddleocr(processor_or_ocr, image_path)
            else:
                pred_label = predict_swapnillo(model, processor_or_ocr, image_path)
            
            is_correct = pred_label.strip() == true_label.strip()
            
            if is_correct:
                correct += 1
            
            predictions.append({
                'image': image_path,
                'true': true_label,
                'pred': pred_label,
                'correct': is_correct
            })
            
            if len(samples) < 10:
                samples.append({
                    'image': image_path,
                    'true': true_label,
                    'pred': pred_label,
                    'correct': is_correct
                })
        except Exception as e:
            print(f"Error on {image_path}: {e}")
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, samples, predictions

def generate_comparison_report(results, output_path):
    """Generate comprehensive comparison report."""
    
    report = f"""# Comprehensive Bangla OCR Evaluation Report

## Overview
This report compares three OCR approaches for Bangla handwritten character recognition:
1. **Baseline Swapnillo**: Pre-trained `swapnillo/Bangla-OCR-SFT` (Qwen3-VL)
2. **Fine-tuned Swapnillo**: Ekush-adapted version with QLoRA
3. **PaddleOCR**: Traditional OCR engine

## Dataset
- **Source**: Ekush handwritten Bangla characters
- **Test Set**: {results['test_size']} images
- **Character Classes**: Digits, vowels, consonants

## Overall Results

| Model | Accuracy | Status |
|-------|----------|--------|
| **Baseline Swapnillo** | {results['baseline']['accuracy']:.2%} | ✓ Tested |
| **Fine-tuned Swapnillo** | {results['finetuned']['accuracy']:.2%} | ✓ Tested |
| **PaddleOCR** | {results['paddleocr']['accuracy']:.2%} | {results['paddleocr']['status']} |

## Sample Predictions

### Baseline Swapnillo
"""
    
    for i, sample in enumerate(results['baseline']['samples'][:5], 1):
        status = "✅" if sample['correct'] else "❌"
        report += f"{i}. {status} True: `{sample['true']}`, Predicted: `{sample['pred']}`\n"
    
    report += "\n### Fine-tuned Swapnillo\n"
    for i, sample in enumerate(results['finetuned']['samples'][:5], 1):
        status = "✅" if sample['correct'] else "❌"
        report += f"{i}. {status} True: `{sample['true']}`, Predicted: `{sample['pred']}`\n"
    
    report += "\n### PaddleOCR\n"
    for i, sample in enumerate(results['paddleocr']['samples'][:5], 1):
        status = "✅" if sample['correct'] else "❌"
        report += f"{i}. {status} True: `{sample['true']}`, Predicted: `{sample['pred']}`\n"
    
    report += f"""

## Analysis

### Baseline Swapnillo
- **Accuracy**: {results['baseline']['accuracy']:.2%}
- **Observation**: Pre-trained model without Ekush-specific adaptation

### Fine-tuned Swapnillo
- **Accuracy**: {results['finetuned']['accuracy']:.2%}
- **Improvement**: {(results['finetuned']['accuracy'] - results['baseline']['accuracy']):.2%}
- **Training**: 3 epochs, QLoRA (r=16, alpha=32)
- **Loss**: Training 0.0, Eval 2.3e-07

### PaddleOCR
- **Accuracy**: {results['paddleocr']['accuracy']:.2%}
- **Status**: {results['paddleocr']['status']}
- **Note**: {results['paddleocr']['note']}

## Conclusion

{results['conclusion']}

## Technical Details
- **VLM Base**: Qwen3-VL (2.1B parameters)
- **Fine-tuning**: QLoRA 4-bit (6.4M trainable params, 0.3%)
- **Training Data**: 900 Ekush images
- **Test Data**: {results['test_size']} images
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nComparison report saved to: {output_path}")

def main():
    model_id = "swapnillo/Bangla-OCR-SFT"
    adapter_path = "/home/node/.openclaw/workspace/BhashaLLM/models/ocr_adapters/final_ocr_adapter"
    test_data_path = "/home/node/.openclaw/workspace/BhashaLLM/data/processed/ocr_training_data.jsonl"
    report_path = "/home/node/.openclaw/workspace/BhashaLLM/report/COMPREHENSIVE_OCR_EVALUATION.md"
    
    # Load test data
    print("Loading test data...")
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Use last 50 samples as test set
    test_data = test_data[-50:]
    print(f"Test set size: {len(test_data)}")
    
    results = {
        'test_size': len(test_data),
        'baseline': {},
        'finetuned': {},
        'paddleocr': {}
    }
    
    # 1. Evaluate Baseline
    print("\n=== Evaluating Baseline Swapnillo ===")
    baseline_model, baseline_processor = load_baseline_model(model_id)
    baseline_acc, baseline_samples, baseline_predictions = evaluate_model(
        "Baseline", baseline_model, baseline_processor, test_data, max_samples=50
    )
    results['baseline'] = {
        'accuracy': baseline_acc,
        'samples': baseline_samples,
        'predictions': baseline_predictions
    }
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    
    # 2. Evaluate Fine-tuned
    print("\n=== Evaluating Fine-tuned Swapnillo ===")
    del baseline_model
    torch.cuda.empty_cache()
    
    finetuned_model, finetuned_processor = load_finetuned_model(model_id, adapter_path)
    finetuned_acc, finetuned_samples, finetuned_predictions = evaluate_model(
        "Fine-tuned", finetuned_model, finetuned_processor, test_data, max_samples=50
    )
    results['finetuned'] = {
        'accuracy': finetuned_acc,
        'samples': finetuned_samples,
        'predictions': finetuned_predictions
    }
    print(f"Fine-tuned Accuracy: {finetuned_acc:.2%}")
    
    # 3. Evaluate PaddleOCR
    print("\n=== Evaluating PaddleOCR ===")
    del finetuned_model
    torch.cuda.empty_cache()
    
    paddleocr, lang = load_paddleocr()
    if paddleocr:
        paddleocr_acc, paddleocr_samples, paddleocr_predictions = evaluate_model(
            "PaddleOCR", None, paddleocr, test_data, max_samples=50
        )
        results['paddleocr'] = {
            'accuracy': paddleocr_acc,
            'samples': paddleocr_samples,
            'predictions': paddleocr_predictions,
            'status': f"✓ Tested ({lang})",
            'note': "Bengali model unavailable, using English" if lang == 'en' else "Using Bengali model"
        }
        print(f"PaddleOCR Accuracy: {paddleocr_acc:.2%}")
    else:
        results['paddleocr'] = {
            'accuracy': 0.0,
            'samples': [],
            'predictions': [],
            'status': "✗ Unavailable",
            'note': "PaddleOCR could not be loaded"
        }
    
    # Generate conclusion
    if finetuned_acc > baseline_acc:
        improvement = finetuned_acc - baseline_acc
        results['conclusion'] = f"The fine-tuned Swapnillo model shows **{improvement:.2%}** improvement over the baseline, demonstrating successful adaptation to Bangla handwritten characters."
    else:
        results['conclusion'] = "The models show similar performance. This may indicate test set distribution issues or need for more diverse training data."
    
    # Summary
    print("\n=== Summary ===")
    print(f"Baseline Swapnillo:    {baseline_acc:.2%}")
    print(f"Fine-tuned Swapnillo:  {finetuned_acc:.2%}")
    print(f"PaddleOCR:             {results['paddleocr']['accuracy']:.2%} ({results['paddleocr']['status']})")
    print(f"Improvement:           {(finetuned_acc - baseline_acc):.2%}")
    
    # Generate report
    generate_comparison_report(results, report_path)

if __name__ == "__main__":
    main()
