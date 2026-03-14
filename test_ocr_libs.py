import easyocr
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import os
import time

def test_ocr_libraries(image_path):
    print(f"Testing OCR libraries on: {image_path}\n")
    print("-" * 50)
    
    # 1. EasyOCR
    print("1. EasyOCR (Bangla)")
    try:
        start_time = time.time()
        # Initialize reader (will download models on first run)
        reader = easyocr.Reader(['bn'])
        result = reader.readtext(image_path, detail=0)
        easy_time = time.time() - start_time
        
        print(f"Time taken: {easy_time:.2f} seconds")
        print("Result:")
        for line in result:
            print(line)
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        
    print("-" * 50)
    
    # 2. PaddleOCR
    print("2. PaddleOCR (Bangla)")
    try:
        start_time = time.time()
        # Initialize PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ar')
        # PaddleOCR doesn't officially support 'bn' OutOfBox in the lightweight models easily, 
        # But let's try 'bn' or default if it exists. 
        # Actually PaddleOCR added bengali support via 'bengali'. Let's use 'bengali'
        paddle_ocr_bn = PaddleOCR(use_angle_cls=True, lang='bengali')
        
        result = paddle_ocr_bn.ocr(image_path, cls=True)
        paddle_time = time.time() - start_time
        
        print(f"Time taken: {paddle_time:.2f} seconds")
        print("Result:")
        if result and result[0]:
            for line in result[0]:
                print(line[1][0]) # Extracting just the text from the result structure
        else:
            print("No text found.")
    except Exception as e:
        print(f"PaddleOCR failed: {e}")

    print("-" * 50)
    
    # 3. Tesseract (PyTesseract)
    print("3. Tesseract (Bangla)")
    try:
        start_time = time.time()
        
        # Tesseract needs the language pack 'ben' installed on the system (e.g. apt-get install tesseract-ocr-ben)
        img = Image.open(image_path)
        result = pytesseract.image_to_string(img, lang='ben')
        tess_time = time.time() - start_time
        
        print(f"Time taken: {tess_time:.2f} seconds")
        print("Result:")
        print(result.strip())
    except Exception as e:
        print(f"Tesseract failed. Did you install 'tesseract-ocr-ben'? Error: {e}")

if __name__ == "__main__":
    test_image = "data/raw/sample_ocr.jpg"
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
    else:
        test_ocr_libraries(test_image)
