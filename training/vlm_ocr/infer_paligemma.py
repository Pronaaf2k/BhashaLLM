#!/usr/bin/env python3
import argparse
import json

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(base_model: str, adapter: str = None):
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def predict(processor, model, image_path: str, prompt: str, max_new_tokens: int):
    image = Image.open(image_path).convert("RGB")
    if not prompt.lstrip().startswith("<image>"):
        prompt = f"<image> {prompt}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run PaliGemma OCR inference.")
    parser.add_argument("--config", default="training/vlm_ocr/config.json")
    parser.add_argument("--image", required=True)
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter, e.g. outputs/.../final_adapter")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processor, model = load_model(cfg["base_model"], args.adapter)
    text = predict(processor, model, args.image, cfg["prompt"], int(cfg.get("max_new_tokens", 256)))
    print(text)


if __name__ == "__main__":
    main()
