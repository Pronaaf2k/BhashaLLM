import sys
import os

# Add local path
sys.path.append(os.getcwd())

from evaluate_ocr_models import compute_metrics

def test_metrics():
    preds = [
        {'true': 'bangla', 'pred': 'bangla', 'correct': True},
        {'true': 'text', 'pred': 'test', 'correct': False},
        {'true': 'ocr', 'pred': 'ocr', 'correct': True}
    ]
    
    cer, wer, class_acc = compute_metrics(preds)
    
    print(f"CER: {cer}")
    print(f"WER: {wer}")
    print(f"Class Acc: {class_acc}")
    
if __name__ == "__main__":
    test_metrics()
