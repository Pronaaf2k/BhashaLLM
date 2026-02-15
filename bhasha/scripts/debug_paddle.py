from paddleocr import PaddleOCR
try:
    # Attempt to access internal dictionaries if available
    from paddleocr.ppocr.utils.utility import initial_logger
    import paddleocr
    print("PaddleOCR version:", paddleocr.__version__)
    
    # Check if we can find language keys
    # Usually in ppocr/utils/gen_label.py or similar, but hard to reach.
    # Let's try to instantiate with a wrong language and catch the error to see if it lists available options
    try:
        PaddleOCR(lang='xyz', show_log=False)
    except Exception as e:
        print("Error when using 'xyz':", e)
        
    # Try 'bn' again with different versions
    print("Trying 'bn' with PP-OCRv4...")
    try:
        PaddleOCR(lang='bn', ocr_version='PP-OCRv4', show_log=False)
        print("Success with PP-OCRv4")
    except Exception as e:
        print("Failed PP-OCRv4:", e)

    print("Trying 'bengali' with PP-OCRv3...")
    try:
        PaddleOCR(lang='bengali', ocr_version='PP-OCRv3', show_log=False)
        print("Success with PP-OCRv3 (bengali)")
    except Exception as e:
        print("Failed PP-OCRv3 (bengali):", e)
        
except ImportError:
    print("Could not import PaddleOCR internals")
