"""
YOLOv8 Medium Detector Training
Config: 300 epochs, batch=16, AdamW lr=0.002, momentum=0.9
Early stopping: patience=50, conf_threshold=0.5
"""
import os
from ultralytics import YOLO

def train_detector():
    # Load YOLOv8 medium model
    model = YOLO('yolov8m.pt')
    
    # Training config
    results = model.train(
        data='data/bangla_chars.yaml',  # Dataset config
        epochs=300,
        batch=16,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.002,
        momentum=0.9,
        patience=50,
        conf=0.5,
        device=0,  # GPU
        workers=8,
        project='models',
        name='yolo_detector'
    )
    
    # Save best model
    model.save('models/yolo_detector.pt')
    print(f"Detector trained. mIoU: {results.box.map}, Precision: {results.box.p}, Recall: {results.box.r}")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train_detector()S