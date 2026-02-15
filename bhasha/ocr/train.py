#!/usr/bin/env python3
"""
Fine-tune swapnillo/Bangla-OCR-SFT (Qwen3-VL) on Ekush dataset.
Uses QLoRA for memory efficiency.
"""
import json
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from pathlib import Path

from bhasha.data.dataset import OCRDataset, collate_fn

from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed" / "banglawriting"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "ocr_adapters" / "banglawriting_adapter"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="swapnillo/Bangla-OCR-SFT")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    args = parser.parse_args()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Check for train/val splits
    data_path = Path(args.data_dir)
    train_path = data_path / "train.jsonl"
    val_path = data_path / "val.jsonl"
    
    # Fallback to single file if splits don't exist (backward compatibility for Ekush)
    if not train_path.exists():
        print(f"Split files not found in {args.data_dir}. Checking for ocr_training_data.jsonl...")
        single_file = Path("/home/node/.openclaw/workspace/BhashaLLM/data/processed/ocr_training_data.jsonl")
        if single_file.exists():
            print(f"Using single file: {single_file}")
            full_dataset = OCRDataset(str(single_file), processor)
            # Create simple split
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        else:
            raise FileNotFoundError(f"No training data found in {args.data_dir} or standard paths.")
    else:
        print(f"Loading training data from: {train_path}")
        print(f"Loading validation data from: {val_path}")
        train_dataset = OCRDataset(str(train_path), processor)
        val_dataset = OCRDataset(str(val_path), processor)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    
    # Load model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Lora Config
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", 
        modules_to_save=None,
    )
    
    # Wrap model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_strategy="steps",
        load_best_model_at_end=False,
        fp16=True, 
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False, 
        gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Model saved to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
