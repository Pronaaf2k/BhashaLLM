#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


class OcrJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, prompt: str):
        self.rows = []
        self.prompt = prompt
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"image": row["image"], "text": row["text"], "prompt": self.prompt}


@dataclass
class PaliGemmaOcrCollator:
    processor: AutoProcessor
    max_length: int

    def _image_prompt(self, prompt: str) -> str:
        return prompt if prompt.lstrip().startswith("<image>") else f"<image> {prompt}"

    def __call__(self, batch: List[dict]):
        prompts = []
        images = []
        labels = []
        for item in batch:
            images.append(Image.open(item["image"]).convert("RGB"))
            prompts.append(self._image_prompt(item["prompt"]))
            labels.append(item["text"])

        encoded = self.processor(
            text=prompts,
            images=images,
            suffix=labels,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "labels" in encoded:
            encoded["labels"][encoded["labels"] == self.processor.tokenizer.pad_token_id] = -100
        else:
            encoded["labels"] = encoded["input_ids"].clone()
            encoded["labels"][encoded["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return encoded


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune PaliGemma 2 for Bangla handwritten OCR.")
    parser.add_argument("--config", default="training/vlm_ocr/config.json")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    args = parser.parse_args()
    cfg = load_config(args.config)

    processor = AutoProcessor.from_pretrained(cfg["base_model"], trust_remote_code=True)
    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg["base_model"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quant_config is not None and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = OcrJsonlDataset(cfg["train_jsonl"], cfg["prompt"])
    val_dataset = OcrJsonlDataset(cfg["val_jsonl"], cfg["prompt"])
    collator = PaliGemmaOcrCollator(processor=processor, max_length=int(cfg.get("max_length", 2048)))

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg.get("epochs", 3)),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 100)),
        eval_steps=int(cfg.get("eval_steps", 100)),
        eval_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=3,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    output_dir = Path(cfg["output_dir"])
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "processor"))


if __name__ == "__main__":
    main()
