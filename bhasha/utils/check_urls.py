import requests

urls = [
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/bn_PP-OCRv3_rec_infer.tar",
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/bn_PP-OCRv4_rec_infer.tar",
    "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bn_mobile_v2.0_rec_infer.tar",
    # Try generic multilingual if specific fails (though BN usually has specific)
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_rec_infer.tar" 
]

for url in urls:
    try:
        response = requests.head(url, timeout=5)
        print(f"{url}: {response.status_code}")
    except Exception as e:
        print(f"{url}: Error {e}")
