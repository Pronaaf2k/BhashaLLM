from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os

import sys

model_id = "swapnillo/Bangla-OCR-SFT"
image_paths = []
if len(sys.argv) > 1:
    image_paths = sys.argv[1:]
else:
    image_paths = ["/home/node/.openclaw/workspace/BhashaLLM/data/raw/sample_ocr.jpg"]

print(f"Loading {model_id}...")
try:
    # Use bfloat16 to save memory
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
        
    print(f"\nProcessing {image_path}...")
    image = Image.open(image_path)

# Standard Qwen-VL prompt structure
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all text from this image."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages) # If needed, but processor usually handles it
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)

print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("\n--- Result ---")
print(output_text[0])
