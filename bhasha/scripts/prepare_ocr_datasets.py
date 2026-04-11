#!/usr/bin/env python3
"""
Download and prepare additional Bangla OCR datasets for improved training.

Datasets:
1. HuggingFace: deepcopy/handwritten-text-recognition-bongabdo
2. Mendeley: BanglaLekha-Isolated (743k6dm543/4)
3. Mendeley: Ekush (r43wkvdk4w/1)  
4. Mendeley: Bangla Handwritten Character Dataset (hf6sf8zrkc/2)
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def download_hf_dataset():
    """Download the HuggingFace handwritten text recognition dataset"""
    print("\n" + "="*60)
    print("Downloading HuggingFace Dataset: deepcopy/handwritten-text-recognition-bongabdo")
    print("="*60)
    
    try:
        dataset = load_dataset("deepcopy/handwritten-text-recognition-bongabdo")
        
        output_dir = PROCESSED_DIR / "bongabdo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train split with 70/15/15 split
        if 'train' in dataset:
            all_data = []
            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            print(f"Processing {len(dataset['train'])} samples...")
            for idx, example in enumerate(tqdm(dataset['train'])):
                image = example['image']
                text = example['text']
                
                # Save image
                image_path = images_dir / f"{idx}.png"
                image.save(image_path)
                
                # Create entry
                all_data.append({
                    "image": str(image_path.relative_to(BASE_DIR)),
                    "text": text
                })
            
            # 70/15/15 split
            import random
            random.seed(42)
            random.shuffle(all_data)
            
            n = len(all_data)
            train_size = int(0.7 * n)
            val_size = int(0.15 * n)
            
            train_data = all_data[:train_size]
            val_data = all_data[train_size:train_size+val_size]
            test_data = all_data[train_size+val_size:]
            
            # Save splits
            for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                with open(output_dir / f"{split_name}.jsonl", 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ Saved Bongabdo dataset:")
            print(f"   Train: {len(train_data)} samples")
            print(f"   Val:   {len(val_data)} samples")
            print(f"   Test:  {len(test_data)} samples")
            return True
    
    except Exception as e:
        print(f"❌ Failed to download HuggingFace dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_existing_ekush():
    """Prepare the existing Ekush dataset in better format"""
    print("\n" + "="*60)
    print("Preparing Existing Ekush Dataset")
    print("="*60)
    
    try:
        ekush_dir = PROCESSED_DIR / "ekush_images"
        if not ekush_dir.exists():
            print("⚠️  Ekush dataset not found")
            return False
        
        output_dir = PROCESSED_DIR / "ekush_prepared"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images and their labels
        data = []
        for class_dir in ekush_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            # The directory name is the label
            label = class_dir.name
            
            for img_path in class_dir.glob("*.png"):
                data.append({
                    "image": str(img_path.relative_to(BASE_DIR)),
                    "text": label
                })
        
        # Split into train/val/test (70/15/15)
        import random
        random.seed(42)
        random.shuffle(data)
        
        n = len(data)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        # Save splits
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            with open(output_dir / f"{split_name}.jsonl", 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Prepared Ekush dataset:")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Val:   {len(val_data)} samples")
        print(f"   Test:  {len(test_data)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Failed to prepare Ekush dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_mendeley_instructions():
    """Provide instructions for Mendeley datasets (require manual download)"""
    print("\n" + "="*60)
    print("Mendeley Datasets (Manual Download Required)")
    print("="*60)
    print("\nThese datasets require manual download from Mendeley:")
    print("\n1. BanglaLekha-Isolated (743k6dm543/4)")
    print("   URL: https://data.mendeley.com/datasets/743k6dm543/4")
    print("   Description: Large-scale isolated Bangla character dataset")
    print("\n2. Ekush (r43wkvdk4w/1)")
    print("   URL: https://data.mendeley.com/datasets/r43wkvdk4w/1")
    print("   Description: Handwritten Bangla characters")
    print("\n3. Bangla Handwritten Character Dataset (hf6sf8zrkc/2)")
    print("   URL: https://data.mendeley.com/datasets/hf6sf8zrkc/2")
    print("   Description: Comprehensive handwritten Bangla characters")
    print("\nAfter downloading, extract to data/raw/mendeley/")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("BhashaLLM OCR Dataset Preparation")
    print("="*60)
    
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Download HuggingFace dataset
    results['hf_bongabdo'] = download_hf_dataset()
    
    # Prepare existing Ekush
    results['ekush'] = prepare_existing_ekush()
    
    # Show Mendeley instructions
    download_mendeley_instructions()
    
    # Summary
    print("\n" + "="*60)
    print("DATASET PREPARATION SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Download Mendeley datasets manually (if needed)")
    print("2. Run: python3 bhasha/ocr/train.py")
    print("3. Evaluate: python3 bhasha/scripts/evaluate_swapnillo.py")
    print("="*60)

if __name__ == "__main__":
    main()
