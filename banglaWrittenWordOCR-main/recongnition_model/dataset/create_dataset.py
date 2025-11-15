# --- START OF FILE create_dataset.py (FIXED) ---

import pandas as pd
import os
from PIL import Image
import numpy as np
import torch
import random

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms

BATCH_SIZE = 64
NUM_CLASSES = {'grapheme_root': 178, 'vowel_diacritic': 11, 'consonant_diacritic': 8}

# Get the directory of the current script (for robust pathing)
# This assumes the 'data' folder is in the parent directory of 'dataset'
# BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') 
# Since your original code uses relative paths like "data/BanglaGrapheme/", 
# we'll stick to that but add an os.path.abspath() for better debugging visibility.

class BanglaGraphemeDataset(Dataset): # Inherit from Dataset
    def __init__(self,img_H,img_W,type):
        self.t = type
        self.width = img_W
        self.height = img_H
        
        # Base path for the images and CSVs (relative to the execution directory)
        BASE_DATA_PATH = "data/BanglaGrapheme"
        
        # Load all data from the CSV
        df = pd.read_csv(f"{BASE_DATA_PATH}/{type}.csv")
        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
        
        # --- FIX IMPLEMENTATION START ---
        
        image_folder = f"{BASE_DATA_PATH}/{self.t}" 
        valid_indices = []
        missing_count = 0
        
        # Iterate through all image IDs loaded from the CSV
        for i, image_id in enumerate(df.image_id.values):
            image_path = os.path.join(image_folder, f"{image_id}.jpg")
            
            # Check if the file exists
            if os.path.exists(image_path):
                valid_indices.append(i)
            else:
                missing_count += 1
                # Optional: Uncomment the print statement to see which files are missing
                # print(f"Warning: Missing file skipped: {image_path}")

        # Filter the arrays to only include existing images
        self.image_ids = df.iloc[valid_indices].image_id.values
        self.grapheme_root = df.iloc[valid_indices].grapheme_root.values
        self.vowel_diacritic = df.iloc[valid_indices].vowel_diacritic.values
        self.consonant_diacritic = df.iloc[valid_indices].consonant_diacritic.values
        
        print(f"Dataset '{type}' initialized. Loaded {len(self.image_ids)} images. Skipped {missing_count} missing files.")

        # --- FIX IMPLEMENTATION END ---
        
    def __len__(self):
        return len(self.image_ids)    
    
    def __getitem__(self,item):
        # The file existence is already guaranteed by __init__
        image_folder = f"data/BanglaGrapheme/{self.t}" 
        image_path = os.path.join(image_folder, f"{self.image_ids[item]}.jpg")
        
        # Since the file exists, this line should now succeed without a FileNotFoundError
        image = Image.open(image_path) 
        
        image = image.resize((self.width,self.height))
        image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        
        image = transform(image)
        return {
            'image': image,
            'grapheme_root': torch.tensor(self.grapheme_root[item],dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[item],dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[item],dtype=torch.long)
        }

    
def dividing_datasets(h, w):
    train = BanglaGraphemeDataset(img_H=h,img_W=w,type='train')
    test = BanglaGraphemeDataset(img_H=h,img_W=w,type='test')
    val = BanglaGraphemeDataset(img_H=h,img_W=w,type='val')

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader