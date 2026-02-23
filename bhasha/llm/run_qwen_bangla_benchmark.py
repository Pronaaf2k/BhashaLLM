import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

QUESTIONS = [
    {
        "category": "Spelling Correction",
        "prompt": "নিচের ভুল বানানের শব্দটি শুদ্ধ করে লিখুন: 'দূরারোগ্য'"
    },
    {
        "category": "Grammar Checking",
        "prompt": "নিচের বাক্যটিতে কোনো ভাষার ভুল থাকলে সংশোধন করুন: 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।'"
    },
    {
        "category": "Word Validation",
        "prompt": "এই শব্দটি কি একটি সঠিক বাংলা শব্দ? 'রোবট'"
    },
    {
        "category": "Metaphor vs Literal",
        "prompt": "নিচের বাক্যটি আক্ষরিক (Literal) নাকি রূপক (Metaphorical) তা নির্ণয় করুন: 'অতিভক্তি চোরের লক্ষণ'"
    }
]

MD_FILEPATH = "/home/benaaf/Desktop/BhashaLLM/llm outputs/qwen_bangla.md"

def main():
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "/home/benaaf/Desktop/BhashaLLM/models/instruct_adapters/final_instruct_adapter"
    
    print(f"Loading base model {base_model_id} from Hugging Face...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    results = "# Qwen Bangla (Qwen2.5-1.5B-Instruct + Fine-tuned Adapter)\n\n"
    results += "| Metric | Result |\n"
    results += "| :--- | :--- |\n"
    results += "| Spelling Correction |  |\n"
    results += "| Grammar Checking |  |\n"
    results += "| Word Validation |  |\n"
    results += "| Metaphor vs Literal |  |\n"
    results += "| Verdict | **Pending Manual Review** |\n\n\n"
    results += "## Benchmark Outputs\n\n"

    for q in QUESTIONS:
        print(f"Running Category: {q['category']}")
        
        messages = [
            {"role": "user", "content": q['prompt']}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.3, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse output accurately. Usually output after chat template format contains prompt + generation.
        # So we split by the prompt text roughly
        prompt_content_stripped = q['prompt'].strip()
        if prompt_content_stripped in generated_text:
            response = generated_text[generated_text.rfind(prompt_content_stripped)+len(prompt_content_stripped):].strip()
            # If Qwen's specific token system added system/user tags we might need to strip those
            if "assistant\n" in response:
                response = response.split("assistant\n")[1].strip()
            # Also let's try direct decode of only the new tokens:
        else:
            response = generated_text
            
        # Safer extraction: Decode only generated tokens
        input_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
        time_taken = round(end_time - start_time, 2)
        results += f"### {q['category']}\n\n"
        results += f"**Prompt:**\n> {q['prompt'].replace(chr(10), chr(10) + '> ')}\n\n"
        results += f"**Response:** ({time_taken}s)\n\n{response}\n\n---\n"
        
        print(f"Completed {q['category']} in {time_taken}s")

    with open(MD_FILEPATH, "w", encoding="utf-8") as f:
         f.write(results)
         
    print(f"Results written to {MD_FILEPATH}")

if __name__ == '__main__':
    main()
