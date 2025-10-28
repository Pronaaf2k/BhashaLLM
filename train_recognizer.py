"""
EfficientNet-B4 Recognition Model Training
Input: 128x256 patches, normalize [-1,1]
Outputs: 3 heads (root:168, vowel:11, consonant:7)
Batch: 64-128, Adam lr=0.01, ReduceLROnPlateau
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4
import json

class BanglaRecognizer(nn.Module):
    def __init__(self, num_roots=168, num_vowels=11, num_consonants=7):
        super().__init__()
        self.backbone = efficientnet_b4(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        self.dropout = nn.Dropout(0.5)
        self.fc_root = nn.Linear(1792, num_roots)
        self.fc_vowel = nn.Linear(1792, num_vowels)
        self.fc_consonant = nn.Linear(1792, num_consonants)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        root = self.fc_root(x)
        vowel = self.fc_vowel(x)
        consonant = self.fc_consonant(x)
        return root, vowel, consonant

def custom_loss(outputs, targets):
    """Average CrossEntropy across 3 outputs"""
    root_out, vowel_out, cons_out = outputs
    root_tgt, vowel_tgt, cons_tgt = targets
    
    loss_root = nn.CrossEntropyLoss()(root_out, root_tgt)
    loss_vowel = nn.CrossEntropyLoss()(vowel_out, vowel_tgt)
    loss_cons = nn.CrossEntropyLoss()(cons_out, cons_tgt)
    
    return (loss_root + loss_vowel + loss_cons) / 3

def train_recognizer(train_loader, val_loader, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BanglaRecognizer().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, (roots, vowels, consonants) in train_loader:
            images = images.to(device)
            roots, vowels, consonants = roots.to(device), vowels.to(device), consonants.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = custom_loss(outputs, (roots, vowels, consonants))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct_root = correct_vowel = correct_cons = total = 0
        
        with torch.no_grad():
            for images, (roots, vowels, consonants) in val_loader:
                images = images.to(device)
                roots, vowels, consonants = roots.to(device), vowels.to(device), consonants.to(device)
                
                outputs = model(images)
                loss = custom_loss(outputs, (roots, vowels, consonants))
                val_loss += loss.item()
                
                # Accuracy
                pred_root = outputs[0].argmax(1)
                pred_vowel = outputs[1].argmax(1)
                pred_cons = outputs[2].argmax(1)
                
                correct_root += (pred_root == roots).sum().item()
                correct_vowel += (pred_vowel == vowels).sum().item()
                correct_cons += (pred_cons == consonants).sum().item()
                total += roots.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        acc_root = correct_root / total
        acc_vowel = correct_vowel / total
        acc_cons = correct_cons / total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Acc - Root:{acc_root:.4f}, Vowel:{acc_vowel:.4f}, Cons:{acc_cons:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/efficientnet_recog.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 6:
                print("Early stopping triggered")
                break

if __name__ == '__main__':
    # Load your dataset here
    # train_dataset = BanglaCharDataset('data/train')
    # val_dataset = BanglaCharDataset('data/val')
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=64)
    # train_recognizer(train_loader, val_loader)
    print("Dataset loading code needed. See comments.")