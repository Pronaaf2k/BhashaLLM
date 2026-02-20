#!/usr/bin/env python3
"""
Train OCR model on combined Bangla datasets with improved configuration.
Combines: Bongabdo (text), Ekush (characters), Banglawriting (characters)
"""

import json
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import argparse
from pathlib import Path

from bhasha.data.dataset import OCRDataset, collate_fn
from model_paths import get_ocr_model_path
import sys

# Configuration - Fix BASE_DIR
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "models" / "ocr_adapters" / "combined_ocr_adapter"

def load_combined_datasets(processor):
    """Load and combine all available OCR datasets"""
    datasets = []
    dataset_info = []
    
    # Dataset paths - SKIP Bongabdo (full sentences, causes tokenization issues)
    # Focus on character-level datasets which are better for OCR
    data_sources = [
        # ("Bongabdo", BASE_DIR / "data" / "processed" / "bongabdo"),  # Skip - full sentences
        ("Ekush", BASE_DIR / "data" / "processed" / "ekush_prepared"),
        ("Banglawriting", BASE_DIR / "data" / "processed" / "banglawriting"),
    ]
    
    train_datasets = []
    val_datasets = []
    
    for name, path in data_sources:
        train_file = path / "train.jsonl"
        val_file = path / "val.jsonl"
        
        if train_file.exists():
            train_ds = OCRDataset(str(train_file), processor)
            train_datasets.append(train_ds)
            dataset_info.append(f"  {name}: {len(train_ds)} train samples")
            
            if val_file.exists():
                val_ds = OCRDataset(str(val_file), processor)
                val_datasets.append(val_ds)
            else:
                print(f"  ⚠️  No validation set for {name}")
        else:
            print(f"  ⚠️  {name} not found at {train_file}")
    
    if not train_datasets:
        raise ValueError("No training datasets found!")
    
    # Combine datasets
    combined_train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    combined_val = ConcatDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)
    
    print("\n📊 Combined Dataset Info:")
    for info in dataset_info:
        print(info)
    print(f"\n  Total Train: {len(combined_train)}")
    if combined_val:
        print(f"  Total Val:   {len(combined_val)}")
    
    return combined_train, combined_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None, help="Base model ID (default: use local)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()
    
    # Use local model if not specified
    if args.model_id is None:
        args.model_id = get_ocr_model_path()
        print(f"Using local model: {args.model_id}")
    
    print("\n" + "="*60)
    print("BhashaLLM OCR Training - Combined Datasets")
    print("="*60)
    
    # Load processor
    print(f"\n📥 Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Load combined datasets
    print("\n📚 Loading datasets...")
    train_dataset, val_dataset = load_combined_datasets(processor)
    
    # Load model in 4-bit
    print(f"\n🤖 Loading model in bf16...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Skip kbit preparation
    # model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration - IMPROVED
    print(f"\n⚙️  Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments - IMPROVED
    print(f"\n🏋️  Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,  # Regularization
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"   Total train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"   Total val samples: {len(val_dataset)}")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {args.output_dir}")
    print(f"\n🧪 To evaluate, run:")
    print(f"   python3 bhasha/scripts/evaluate_swapnillo.py --adapter_path {args.output_dir}")

if __name__ == "__main__":
    main()
