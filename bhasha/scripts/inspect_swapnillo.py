from transformers import AutoConfig
from huggingface_hub import hf_hub_download
import json

model_id = "swapnillo/Bangla-OCR-SFT"

print(f"Inspecting {model_id}...")
try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("Architectures:", config.architectures)
    print("Model Type:", config.model_type)
    print("Config:", config)
except Exception as e:
    print(f"Error loading config: {e}")
