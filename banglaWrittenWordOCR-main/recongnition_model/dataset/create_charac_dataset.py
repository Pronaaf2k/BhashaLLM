import pandas as pd
import os
from PIL import Image
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

BATCH_SIZE = 64

class BanglaGraphemeDataset:
    def __init__(self,img_H,img_W,type):
        df = pd.read_csv(f"data/BanglaGrapheme/{type}.csv")
        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
        
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        
        self.width = img_W
        self.height = img_H
        self.t = type
        
    def __len__(self):
        return len(self.image_ids)    
    
    def __getitem__(self,item):
        image_folder = f"data/BanglaGrapheme/{self.t}" 
        image_path = os.path.join(image_folder, f"{self.image_ids[item]}.jpg")
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

    
def extract_foldername(image_id):
    return image_id.split('_')[0]    

def dividing_datasets(h, w):
    train = BanglaGraphemeDataset(img_H=h,img_W=w,type='train')
    test = BanglaGraphemeDataset(img_H=h,img_W=w,type='test')
    val = BanglaGraphemeDataset(img_H=h,img_W=w,type='val')

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader
