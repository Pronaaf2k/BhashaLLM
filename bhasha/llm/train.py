import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, Dataset

from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "adapters"
LOG_DIR = BASE_DIR / "logs"

def train(args):
    print(f"Loading model: {args.model_name}")
    
    # 1. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # usually right for training, left for generation

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        # Target modules vary by architecture. For Qwen/Llama these are standard. 
        # For BLOOM: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load & Tokenize Dataset
    print("Loading datasets...")
    # We load text files. For CLM, we usually want blocks of context_length.
    def load_text_file(split):
        path = os.path.join(DATA_DIR, f"{split}.txt")
        return load_dataset("text", data_files={"train": path})['train']

    train_ds = load_text_file("train")
    val_ds = load_text_file("val")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.context_length, padding="max_length")

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Training Arguments
    run_name = f"bangla_adapt_{args.model_name.split('/')[-1]}"
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, run_name),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        logging_dir=LOG_DIR,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, run_name, "final_adapter"))
    
    # Also save loss log
    log_history = trainer.state.log_history
    import json
    with open(os.path.join(LOG_DIR, f"{run_name}_history.json"), 'w') as f:
        json.dump(log_history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name")
    parser.add_argument("--batch_size", type=int, default=1) # Reduced for 200W constraint
    parser.add_argument("--context_length", type=int, default=1024) # Reduced for safety
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    
    train(args)
