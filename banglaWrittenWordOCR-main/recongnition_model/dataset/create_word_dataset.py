import pandas as pd
import os
from PIL import Image, ImageOps
import torch
import ast

from torch.utils.data import DataLoader
from torchvision import transforms

BATCH_SIZE = 64
IMG_H = 128
IMG_W = 128

class BanglaWordDataset:
    def __init__(self,type):
        df = pd.read_csv(f"data/BanglaWords/{type}.csv")
        df = df[['image_ids', 'grapheme_root', 'vowel_diacritics', 'consonant_diacritics']]
        
        self.image_ids = df.image_ids.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritics.values
        self.consonant_diacritic = df.consonant_diacritics.values
        self.height = IMG_H
        self.t = type
        
    def __len__(self):
        return len(self.image_ids)    
    
    def __getitem__(self,item):
        
        g = ast.literal_eval(self.grapheme_root[item])
        v = ast.literal_eval(self.vowel_diacritic[item])
        c = ast.literal_eval(self.consonant_diacritic[item])
        
        width = len(g)*IMG_W
        
        image_folder = f"data/BanglaWords/{self.t}" 
        if self.t == 'test':
            image_path = os.path.join(image_folder, f"{self.image_ids[item]}.jpg")
            image = Image.open(image_path)
        else:
            image_path = os.path.join(image_folder, f"{self.image_ids[item]}.bmp")  
            image = Image.open(image_path)
            image = ImageOps.invert(image)
        image = image.resize((width,self.height))
        image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        
        image = transform(image)
        return {
            'image': image,
            'grapheme_root': torch.tensor(g,dtype=torch.long),
            'vowel_diacritic': torch.tensor(v,dtype=torch.long),
            'consonant_diacritic': torch.tensor(c,dtype=torch.long)
        }
        
def dividing_datasets():
    train = BanglaWordDataset(type='train')
    test = BanglaWordDataset(type='test')
    val = BanglaWordDataset(type='val')

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader