#!/usr/bin/env python3
import os
import json
from pathlib import Path
import sys

# Try to import datasets, if it fails give instructions
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets library: pip install datasets")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "bn_htrd"

def main():
    print(f"Downloading BN-HTRd dataset from HuggingFace mirror...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the document-level handwritten text recognition dataset
    # This is a mirror of the Mendeley 743k6dm543/4 dataset 
    dataset = load_dataset("shaoncsecu/BN-HTRd_Splitted", trust_remote_code=True)
    
    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            continue
            
        print(f"Processing {split} split ({len(dataset[split])} samples)...")
        # Map HuggingFace 'validation' name to our standard 'val' filename
        out_name = 'val.jsonl' if split == 'validation' else f"{split}.jsonl"
        out_path = PROCESSED_DIR / out_name
        
        img_dir = PROCESSED_DIR / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(dataset[split]):
                image = item['image']
                text = item['text']
                
                # Save image
                img_path = img_dir / f"{idx}.png"
                image.save(img_path)
                
                # Write standard JSONL entry
                entry = {
                    "image": str(img_path.relative_to(BASE_DIR)),
                    "text": text
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        print(f"✅ Saved {out_name}")

if __name__ == "__main__":
    main()
