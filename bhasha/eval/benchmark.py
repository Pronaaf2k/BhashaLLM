import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import argparse
import os

# Benchmark Config
MODELS = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

TASKS = {
    "translation_en_bn": {
        "prompt_template": "Translate the following English text to Bangla.\n\nEnglish: {input}\n\nBangla:",
        "samples": [
            {"input": "Artificial Intelligence is transforming the world.", "target": "কৃত্রিম বুদ্ধিমত্তা বিশ্বকে বদলে দিচ্ছে।"},
            {"input": "The weather is very nice today.", "target": "আজকের আবহাওয়া খুব সুন্দর।"}
        ]
    },
    "summarization": {
        "prompt_template": "Summarize the following Bangla text in one sentence.\n\nText: {input}\n\nSummary:",
        "samples": [
            {"input": "বাংলাদেশের ভূপ্রকৃতি অত্যন্ত বৈচিত্র্যময়। এখানে অসংখ্য নদ-নদী জালের মতো ছড়িয়ে আছে। বর্ষাকালে এই নদীগুলো পানিতে কানায় কানায় পূর্ণ হয়ে ওঠে। কৃষিই এদেশের অর্থনীতির প্রধান চালিকাশক্তি।", "target": "নদীমাতৃক বাংলাদেশের অর্থনীতি মূলত কৃষিনির্ভর।"}
        ]
    },
    "creative_writing": {
        "prompt_template": "Write a short poem in Bangla about '{input}'.\n\nPoem:",
        "samples": [
            {"input": "বৃষ্টি (Rain)", "target": "(Open ended)"}
        ]
    },
    "ocr_correction": {
        "prompt_template": "Correct the OCR errors in this Bangla text.\n\nInput: {input}\n\nCorrected:",
        "samples": [
            {"input": "বাংলা ভাষা একটি ইন্দা-আর্য ভাষা", "target": "বাংলা ভাষা একটি ইন্দো-আর্য ভাষা"},
            {"input": "সূর্য পূর্ব দিকে উথে", "target": "সূর্য পূর্ব দিকে ওঠে"}
        ]
    },
    "qa": {
        "prompt_template": "Answer in Bangla.\n\nContext: {context}\nQuestion: {question}\n\nAnswer:",
        "samples": [
            {"context": "বাংলাদেশের রাজধানী ঢাকা।", "question": "বাংলাদেশের রাজধানী কী?", "target": "ঢাকা"}
        ]
    }
}

def load_model(model_name, quantize=False):
    print(f"Loading {model_name}...")
    
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if not quantize else None
        )
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None, None

def run_benchmark(model_key, quantize=False):
    model_path = MODELS.get(model_key, model_key)
    tokenizer, model = load_model(model_path, quantize)
    
    if not model:
        return

    results = []
    
    print(f"Running benchmarks on {model_key}...")
    
    for task_name, task_data in TASKS.items():
        print(f"  Task: {task_name}")
        prompt_template = task_data["prompt_template"]
        
        for sample in tqdm(task_data["samples"]):
            # Format prompt
            if task_name == "qa":
                prompt = prompt_template.format(context=sample["context"], question=sample["question"])
            else:
                prompt = prompt_template.format(input=sample["input"])
            
            # Chat template formatting if supported (for Instruct models)
            messages = [{"role": "user", "content": prompt}]
            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                input_text = prompt # Fallback

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    temperature=0.1, 
                    do_sample=True
                )
            end_time = time.time()
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "model": model_key,
                "task": task_name,
                "input": sample.get("input", sample.get("question")),
                "output": generated_text.strip(),
                "target": sample.get("target"),
                "time": round(end_time - start_time, 3)
            })

    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    filename = f"benchmark_results/{model_key}_results.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangla LLM Benchmark")
    parser.add_argument("--model", type=str, default="gemma-2b", help="Model key (gemma-2b, llama-3.2-1b, etc)")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    run_benchmark(args.model, args.quantize)
