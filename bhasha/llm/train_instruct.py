import os
import torch
import argparse
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configuration
OUTPUT_DIR = "/home/node/.openclaw/workspace/BhashaLLM/models/instruct_adapters"
LOG_DIR = "/home/node/.openclaw/workspace/BhashaLLM/logs"
DATA_PATH = "/home/node/.openclaw/workspace/BhashaLLM/data/processed/synthetic_grading.jsonl"

def train(args):
    print(f"Loading base model: {args.base_model}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # Load adapter for continued training
    if args.adapter_path:
        print(f"Loading adapter for continued training from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        # Ensure we are in training mode
        model.print_trainable_parameters()
    else:
        # Should not happen in this phase, but fallback
        print("No adapter path provided. Initializing new LoRA config.")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset('json', data_files=DATA_PATH, split='train')
    train_test = dataset.train_test_split(test_size=0.1)
    
    print(f"Train size: {len(train_test['train'])}, Test size: {len(train_test['test'])}")

    print("Starting SFT...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        max_length=args.context_length,
        dataset_text_field="text",
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none", # Disable tensorboard
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_test['train'],
        eval_dataset=train_test['test'],
        # No peft_config here, as model is already a PeftModel
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_instruct_adapter"))
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_instruct_adapter')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--context_length", type=int, default=512)
    args = parser.parse_args()
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    train(args)
