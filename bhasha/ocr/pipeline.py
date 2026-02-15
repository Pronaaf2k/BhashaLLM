import cv2
import numpy as np
import argparse
import sys
from paddleocr import PaddleOCR
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Configuration
# Initialize PaddleOCR once
# Initialize PaddleOCR with specific version
# Initialize PaddleOCR with specific version
try:
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='bn', show_log=False)
except Exception as e:
    print(f"Warning: PaddleOCR initialization failed: {e}")
    ocr_engine = None

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding (good for handwritten/noisy text)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Deskewing (Simple projection profile based or moments)
    # For now, simple moments based deskew
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def run_paddle(img):
    if ocr_engine is None:
        return "", 0.0
    result = ocr_engine.ocr(img, cls=True)
    text = ""
    scores = []
    if result and result[0]:
        for line in result[0]:
            text += line[1][0] + " "
            scores.append(line[1][1])
    return text.strip(), np.mean(scores) if scores else 0.0

def run_tesseract(img):
    if pytesseract is None:
        return "", 0.0
    try:
        data = pytesseract.image_to_data(img, lang='ben', output_type=pytesseract.Output.DICT)
        text = " ".join([x for x in data['text'] if x.strip()])
        confs = [int(x) for x in data['conf'] if x != '-1']
        return text.strip(), np.mean(confs)/100.0 if confs else 0.0
    except Exception as e:
        print(f"Tesseract error: {e}")
        return "", 0.0

def ensemble_ocr(img):
    # Run both
    p_text, p_conf = run_paddle(img)
    t_text, t_conf = run_tesseract(img)
    
    print(f"Paddle: {p_text} (Conf: {p_conf:.2f})")
    print(f"Tesseract: {t_text} (Conf: {t_conf:.2f})")
    
    # Logic: Prefer non-empty result with higher confidence
    if not p_text and not t_text:
        return ""
    if not p_text:
        return t_text
    if not t_text:
        return p_text
        
    if p_conf >= t_conf:
        return p_text
    else:
        return t_text

def correct_text_with_llm(text, model_path=None):
    # Stub for LLM correction. 
    # In real pipeline, load the finetuned model and prompt it.
    # prompt = f"Correct the following Bangla OCR text:\n{text}\nCorrection:"
    return text # Placeholder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()
    
    try:
        preprocessed = preprocess_image(args.image)
        # cv2.imwrite("debug_preprocessed.png", preprocessed)
        
        final_text = ensemble_ocr(preprocessed)
        print("-" * 30)
        print("Final Ensembled Text:")
        print(final_text)
        
        corrected = correct_text_with_llm(final_text)
        print("-" * 30)
        print("LLM Corrected Text (Stub):")
        print(corrected)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
