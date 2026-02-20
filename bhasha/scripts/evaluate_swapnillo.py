import argparse
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import sys
from peft import PeftModel

def main():
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    parser = argparse.ArgumentParser(description="Evaluate OCR model (Baseline or Fine-tuned)")
    parser.add_argument("--model_id", type=str, default="swapnillo/Bangla-OCR-SFT", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--image_path", type=str, default=str(BASE_DIR / "data" / "raw" / "sample_ocr.jpg"), help="Path to image")
    args = parser.parse_args()

    print(f"Loading base model: {args.model_id}...")
    try:
        # Use bfloat16 to save memory
        # Try Qwen2VL specific class first, then generic
        try: 
            model_class = Qwen2VLForConditionalGeneration
        except NameError:
            from transformers import AutoModel
            model_class = AutoModel

        model = model_class.from_pretrained(
            args.model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        
        if args.adapter_path:
            print(f"Loading adapter from: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    image_paths = [args.image_path]
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
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract all text from this image."}
                ]
            }
        ]
    
        # PREPARE INPUTS CORRECTLY
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # CRITICAL FIX: Get actual processed image tensors
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs, 
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    
        print("Generating...")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
        print(f"\n--- Result for {image_path} ---")
        print(output_text[0])

if __name__ == "__main__":
    main()
