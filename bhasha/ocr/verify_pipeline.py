from paddleocr import PaddleOCR
import os

def test_ocr():
    image_path = "/home/node/.openclaw/workspace/BhashaLLM/data/raw/sample_ocr.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print("Initializing PaddleOCR (English check)...")
    try:
        ocr_en = PaddleOCR(use_angle_cls=True, lang='en')
        print("English OCR initialized.")
    except Exception as e:
        print(f"English OCR failed: {e}")

    print("Initializing PaddleOCR (Bengali check - Explicit Model Names)...")
    try:
        # Bypassing the language check by providing model names directly
        # Attempt 1: bn_PP-OCRv3_mobile_rec
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv3_mobile_det",
            text_recognition_model_name="bn_PP-OCRv3_mobile_rec"
        )
        
        print(f"Running OCR on {image_path}...")
        result = ocr.ocr(image_path, cls=True)
        
        print("\nOCR Result:")
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    print(line)
            else:
                print("No text detected.")

    except Exception as e:
        print(f"PaddleOCR Explicit Init failed: {e}")

if __name__ == "__main__":
    test_ocr()
