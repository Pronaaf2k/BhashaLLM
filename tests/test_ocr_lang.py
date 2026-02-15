from paddleocr import PaddleOCR
import sys

def test_lang(lang_code):
    print(f"Testing language code: {lang_code}")
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, use_gpu=False, show_log=False)
        print(f"SUCCESS: {lang_code}")
        return True
    except Exception as e:
        print(f"FAILED: {lang_code}")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    candidates = ['bn', 'bengali', 'ben']
    for lang in candidates:
        if test_lang(lang):
            break
