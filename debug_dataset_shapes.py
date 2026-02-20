#!/usr/bin/env python3
"""
Debug script to check actual tensor shapes from our dataset
"""

from bhasha.data.dataset import OCRDataset
from transformers import AutoProcessor
from model_paths import get_ocr_model_path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load processor
model_id = get_ocr_model_path()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Load a small sample from each dataset
datasets = [
    ("Bongabdo", BASE_DIR / "data" / "processed" / "bongabdo" / "train.jsonl"),
    ("Ekush", BASE_DIR / "data" / "processed" / "ekush_prepared" / "train.jsonl"),
    ("Banglawriting", BASE_DIR / "data" / "processed" / "banglawriting" / "train.jsonl"),
]

for name, path in datasets:
    if path.exists():
        print(f"\n{'='*60}")
        print(f"Testing {name} dataset")
        print(f"{'='*60}")
        
        dataset = OCRDataset(str(path), processor)
        
        # Get first 3 samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  input_ids shape: {sample['input_ids'].shape}")
            print(f"  pixel_values shape: {sample['pixel_values'].shape if sample['pixel_values'] is not None else 'None'}")
            print(f"  image_grid_thw shape: {sample['image_grid_thw'].shape if sample['image_grid_thw'] is not None else 'None'}")
            print(f"  labels shape: {sample['labels'].shape}")
