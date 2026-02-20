#!/usr/bin/env python3
"""
Helper script to get the correct local model paths.
This ensures all scripts use the local base models instead of downloading from HuggingFace.
"""

from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Local base model paths
QWEN_MODEL_PATH = BASE_DIR / "models" / "base_models" / "Qwen2.5-1.5B-Instruct" / "snapshots"
OCR_MODEL_PATH = BASE_DIR / "models" / "base_models" / "Bangla-OCR-SFT" / "snapshots"

def get_qwen_model_path():
    """Get the local path to Qwen2.5-1.5B-Instruct model"""
    # Find the snapshot directory (there should be only one)
    snapshots = list(QWEN_MODEL_PATH.glob("*"))
    if snapshots:
        return str(snapshots[0])
    # Fallback to HuggingFace if local not found
    return "Qwen/Qwen2.5-1.5B-Instruct"

def get_ocr_model_path():
    """Get the local path to Bangla-OCR-SFT model"""
    # Find the snapshot directory (there should be only one)
    snapshots = list(OCR_MODEL_PATH.glob("*"))
    if snapshots:
        return str(snapshots[0])
    # Fallback to HuggingFace if local not found
    return "swapnillo/Bangla-OCR-SFT"

def get_adapter_path(adapter_type):
    """Get the path to a fine-tuned adapter"""
    adapter_paths = {
        "bangla_llm": BASE_DIR / "models" / "bangla_adapters" / "final_adapter",
        "instruct": BASE_DIR / "models" / "instruct_adapters" / "final_instruct_adapter",
        "ocr": BASE_DIR / "models" / "ocr_adapters" / "banglawriting_adapter",
        "ocr_final": BASE_DIR / "models" / "ocr_adapters" / "final_ocr_adapter",
    }
    return str(adapter_paths.get(adapter_type, ""))

if __name__ == "__main__":
    print("Local Model Paths:")
    print(f"  Qwen2.5-1.5B: {get_qwen_model_path()}")
    print(f"  Bangla-OCR:   {get_ocr_model_path()}")
    print("\nAdapter Paths:")
    print(f"  Bangla LLM:   {get_adapter_path('bangla_llm')}")
    print(f"  Instruct:     {get_adapter_path('instruct')}")
    print(f"  OCR:          {get_adapter_path('ocr')}")
