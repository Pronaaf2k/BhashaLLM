#!/usr/bin/env python3
"""
Download base models from HuggingFace to local cache.
This avoids needing HF tokens and speeds up future runs.
"""

from pathlib import Path
import os

def download_model(model_id, model_type="text"):
    """Download a model from HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"{'='*60}")
    
    try:
        if model_type == "text":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            print("Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Keep on CPU to save memory
                trust_remote_code=True
            )
            
            print(f"✅ Successfully downloaded {model_id}")
            print(f"   Cached at: {tokenizer.name_or_path}")
            return True
            
        elif model_type == "vision":
            from transformers import AutoProcessor
            import torch
            
            print("Downloading processor...")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            print("Downloading model...")
            try:
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True
                )
            except:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True
                )
            
            print(f"✅ Successfully downloaded {model_id}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("BhashaLLM Base Model Downloader")
    print("="*60)
    print("\nThis will download the following base models:")
    print("1. Qwen/Qwen2.5-1.5B-Instruct (for LLM and grading)")
    print("2. swapnillo/Bangla-OCR-SFT (for OCR)")
    print("\nModels will be cached in ~/.cache/huggingface/")
    print("="*60)
    
    results = {}
    
    # Download text models
    print("\n📥 Downloading text models...")
    results['qwen'] = download_model("Qwen/Qwen2.5-1.5B-Instruct", "text")
    
    # Download vision models
    print("\n📥 Downloading vision models...")
    results['ocr'] = download_model("swapnillo/Bangla-OCR-SFT", "vision")
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    all_success = all(results.values())
    print("="*60)
    
    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print("\nModels are now cached locally. Future runs won't need to download them.")
        print("\nYou can now run:")
        print("  python3 test_models.py")
        print("  python3 bhasha/scripts/evaluate_swapnillo.py")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
    
    return all_success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
