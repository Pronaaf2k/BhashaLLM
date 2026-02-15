
from setuptools import setup, find_packages

setup(
    name="bhasha",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "transformers",
        "peft",
        "datasets",
        "pillow",
        "google-generativeai",
        "paddleocr",
        "opencv-python",
        "pandas",
        "scikit-learn",
        "bitsandbytes",
        "qwen_vl_utils"
    ],
)
