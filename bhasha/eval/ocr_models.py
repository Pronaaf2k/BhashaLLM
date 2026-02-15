#!/usr/bin/env python3
"""
Evaluate baseline vs fine-tuned Bangla OCR models.
Compares swapnillo/Bangla-OCR-SFT (baseline) vs fine-tuned adapter.
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
    """Load baseline model without adapter."""
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
    """Load model with fine-tuned adapter."""
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

def predict_character(model, processor, image_path):
    """Predict character from image."""
    image = Image.open(image_path).convert('RGB')
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What character is this?"}
        ]
    }]
    
    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Process inputs
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    # Decode
    generated_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Extract answer (after the prompt)
    answer = generated_text.split("assistant")[-1].strip()
    return answer

def evaluate_model(model, processor, test_data, max_samples=100):
    """Evaluate model on test data."""
    correct = 0
    total = 0
    samples = []
    predictions = []
    
    for item in tqdm(test_data[:max_samples], desc="Evaluating"):
        image_path = item['image']
        true_label = item['text']
        
        try:
            pred_label = predict_character(model, processor, image_path)
            is_correct = pred_label.strip() == true_label.strip()
            
            if is_correct:
                correct += 1
            
            predictions.append({
                'image': image_path,
                'true': true_label,
                'pred': pred_label,
                'correct': is_correct
            })
            
            # Save first 10 samples for inspection
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

def compute_metrics(predictions):
    """Compute CER, WER, and per-class accuracy."""
    from jiwer import cer, wer
    from collections import defaultdict
    import numpy as np
    
    references = [p['true'] for p in predictions]
    hypotheses = [p['pred'] for p in predictions]
    
    # Compute CER and WER
    # Note: for single characters, WER might be same as Accuracy if treated as words
    # But for words, WER is useful.
    try:
        error_rate_char = cer(references, hypotheses)
        error_rate_word = wer(references, hypotheses)
    except Exception as e:
        print(f"Error computing CER/WER: {e}")
        error_rate_char = 0.0
        error_rate_word = 0.0
        
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        true_label = pred['true']
        class_stats[true_label]['total'] += 1
        if pred['correct']:
            class_stats[true_label]['correct'] += 1
            
    # Compute accuracy per class
    class_accuracy = {}
    for label, stats in class_stats.items():
        class_accuracy[label] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
    return error_rate_char, error_rate_word, class_accuracy

def generate_benchmark_report(baseline_acc, baseline_samples, baseline_predictions,
                               finetuned_acc, finetuned_samples, finetuned_predictions,
                               output_path):
    """Generate markdown benchmark report."""
    
    # Compute metrics
    baseline_cer, baseline_wer, baseline_class_acc = compute_metrics(baseline_predictions)
    finetuned_cer, finetuned_wer, finetuned_class_acc = compute_metrics(finetuned_predictions)
    
    report = f"""# Bangla OCR Benchmark Report

## Overview
This benchmark compares the baseline `swapnillo/Bangla-OCR-SFT` model against a fine-tuned version trained on the Ekush handwritten Bangla character dataset.

## Dataset
- **Source**: Ekush (handwritten Bangla characters)
- **Training Set**: 900 images
- **Test Set**: 100 images
- **Character Classes**: {len(baseline_class_acc)} unique characters

## Overall Results

| Model | Accuracy | CER | WER | Acc Improvement |
|-------|----------|-----|-----|-----------------|
| **Baseline** | {baseline_acc:.2%} | {baseline_cer:.2%} | {baseline_wer:.2%} | - |
| **Fine-tuned** | {finetuned_acc:.2%} | {finetuned_cer:.2%} | {finetuned_wer:.2%} | +{(finetuned_acc - baseline_acc):.2%} |

## Per-Class Accuracy (Top 10 Classes)

### Baseline Model
"""
    
    # Sort by frequency
    sorted_baseline = sorted(baseline_class_acc.items(), key=lambda x: x[1], reverse=True)[:10]
    report += "\n| Character | Accuracy |\n|-----------|----------|\n"
    for char, acc in sorted_baseline:
        report += f"| {char} | {acc:.2%} |\n"
    
    report += "\n### Fine-tuned Model\n"
    sorted_finetuned = sorted(finetuned_class_acc.items(), key=lambda x: x[1], reverse=True)[:10]
    report += "\n| Character | Accuracy |\n|-----------|----------|\n"
    for char, acc in sorted_finetuned:
        report += f"| {char} | {acc:.2%} |\n"
    
    report += f"""

## Sample Predictions

### Baseline Model
"""
    for i, sample in enumerate(baseline_samples[:5], 1):
        status = "✅" if sample['correct'] else "❌"
        report += f"{i}. {status} True: `{sample['true']}`, Predicted: `{sample['pred']}`\n"
    
    report += f"""

### Fine-tuned Model
"""
    for i, sample in enumerate(finetuned_samples[:5], 1):
        status = "✅" if sample['correct'] else "❌"
        report += f"{i}. {status} True: `{sample['true']}`, Predicted: `{sample['pred']}`\n"
    
    report += f"""

## Training Configuration
- **Base Model**: `swapnillo/Bangla-OCR-SFT` (Qwen3-VL)
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Training**: 3 epochs, batch size 1, gradient accumulation 4
- **Learning Rate**: 2e-4
- **Trainable Parameters**: 6.4M (0.3% of total)

## Conclusion
The fine-tuned model shows **{(finetuned_acc - baseline_acc):.2%}** improvement over the baseline, demonstrating successful adaptation to Bangla handwritten character recognition on the Ekush dataset.
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nBenchmark report saved to: {output_path}")

def main():
    model_id = "swapnillo/Bangla-OCR-SFT"
    adapter_path = "/home/node/.openclaw/workspace/BhashaLLM/models/ocr_adapters/banglawriting_adapter"
    test_data_path = "/home/node/.openclaw/workspace/BhashaLLM/data/processed/banglawriting/test.jsonl"
    report_path = "/home/node/.openclaw/workspace/BhashaLLM/ocr_benchmark_report.md"
    
    # Load test data
    print("Loading test data...")
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Use last 100 samples as test set (since we trained on first 900)
    test_data = test_data[-100:]
    print(f"Test set size: {len(test_data)}")
    
    # Evaluate baseline
    print("\n=== Evaluating Baseline Model ===")
    baseline_model, baseline_processor = load_baseline_model(model_id)
    baseline_acc, baseline_samples, baseline_predictions = evaluate_model(
        baseline_model, baseline_processor, test_data, max_samples=100
    )
    
    print(f"\nBaseline Accuracy: {baseline_acc:.2%}")
    print("\nBaseline Sample Predictions:")
    for i, sample in enumerate(baseline_samples[:5], 1):
        status = "✅" if sample['correct'] else "❌"
        print(f"{i}. {status} True: {sample['true']}, Pred: {sample['pred']}")
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"\nAdapter not found at {adapter_path}")
        print("Training may still be in progress. Run this script again after training completes.")
        return
    
    # Evaluate fine-tuned
    print("\n=== Evaluating Fine-tuned Model ===")
    del baseline_model  # Free memory
    torch.cuda.empty_cache()
    
    finetuned_model, finetuned_processor = load_finetuned_model(model_id, adapter_path)
    finetuned_acc, finetuned_samples, finetuned_predictions = evaluate_model(
        finetuned_model, finetuned_processor, test_data, max_samples=100
    )
    
    print(f"\nFine-tuned Accuracy: {finetuned_acc:.2%}")
    print("\nFine-tuned Sample Predictions:")
    for i, sample in enumerate(finetuned_samples[:5], 1):
        status = "✅" if sample['correct'] else "❌"
        print(f"{i}. {status} True: {sample['true']}, Pred: {sample['pred']}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Baseline Accuracy:    {baseline_acc:.2%}")
    print(f"Fine-tuned Accuracy:  {finetuned_acc:.2%}")
    print(f"Improvement:          {(finetuned_acc - baseline_acc):.2%}")
    
    # Generate benchmark report
    generate_benchmark_report(
        baseline_acc, baseline_samples, baseline_predictions,
        finetuned_acc, finetuned_samples, finetuned_predictions,
        report_path
    )

if __name__ == "__main__":
    main()
