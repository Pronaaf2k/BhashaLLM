from datasets import load_dataset
import os

def download_sample():
    print("Attempting to load 'z81980440/bengali-synth-ocr'...")
    try:
        ds = load_dataset("z81980440/bengali-synth-ocr", split="train", streaming=True)
        item = next(iter(ds))
        print(f"Loaded item keys: {item.keys()}")
        
        # Save image
        out_path = "sample_ocr.png"
        item['image'].save(out_path)
        print(f"Successfully saved {out_path}")
        
        # Print ground truth if available in keys
        if 'text' in item:
            print(f"Ground Truth Text: {item['text']}")
            
    except Exception as e:
        print(f"Failed to download sample: {e}")

if __name__ == "__main__":
    download_sample()
