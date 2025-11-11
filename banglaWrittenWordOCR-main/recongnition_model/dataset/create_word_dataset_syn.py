import pandas as pd
from PIL import Image
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

import word_mapping as m

BATCH_SIZE = 64
IMG_H = 32
IMG_W = 128

class CustomBanglaDataset:
    def __init__(self, type):
        df = pd.read_csv(f"data/BanglaWords_Syn/{type}.csv")
        df = df[['image_label', 'target']]

        self.image_label = df.image_label.values
        self.target = df.target.values
        self.t = type
        self.mapping = m.mapping

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self,item):

        image_folder = f"data/BanglaWords_Syn/{self.t}" 
        image_path = os.path.join(image_folder, f"{self.image_label[item]}.jpg")
        image = Image.open(image_path)
        image = image.resize((IMG_W,IMG_H))
        image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        
        image = transform(image)
        word = self.target[item]
        word = [char for char in word if char not in ['\u200c', '\"', '-', '\u200d', '\n', '\xa0', '\r']]
        target = [m.mapping.index(char) for char in word]

        
        max_length = 25  
        padded_target = F.pad(torch.tensor(target, dtype=torch.long), (0, max_length - len(target)))


        return {
            'image': image,
            'target': padded_target
        }  

def forming_datasets():
    train = CustomBanglaDataset(type='train')
    test = CustomBanglaDataset(type='test')
    val = CustomBanglaDataset(type='val')

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader