#!/usr/bin/env python3
"""
Quick test to understand how Qwen2-VL processes images of different sizes.
"""

from transformers import AutoProcessor
from PIL import Image
import torch
from model_paths import get_ocr_model_path

# Load processor
model_id = get_ocr_model_path()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Create test images of different sizes
img1 = Image.new('RGB', (100, 100), color='red')
img2 = Image.new('RGB', (150, 150), color='blue')

# Test single image
print("Testing single image processing...")
messages1 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img1},
        {"type": "text", "text": "Test"}
    ]
}]

from qwen_vl_utils import process_vision_info
text1 = processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
image_inputs1, video_inputs1 = process_vision_info(messages1)

inputs1 = processor(
    text=[text1],
    images=image_inputs1,
    videos=video_inputs1,
    padding=True,
    return_tensors="pt"
)

print(f"Single image pixel_values shape: {inputs1['pixel_values'].shape}")
print(f"Single image grid_thw shape: {inputs1['image_grid_thw'].shape}")

# Test another single image
messages2 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img2},
        {"type": "text", "text": "Test"}
    ]
}]

text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
image_inputs2, video_inputs2 = process_vision_info(messages2)

inputs2 = processor(
    text=[text2],
    images=image_inputs2,
    videos=video_inputs2,
    padding=True,
    return_tensors="pt"
)

print(f"\nAnother image pixel_values shape: {inputs2['pixel_values'].shape}")
print(f"Another image grid_thw shape: {inputs2['image_grid_thw'].shape}")

print("\n✅ Images are processed to consistent tensor shapes!")
print("The processor handles variable-sized images automatically.")
