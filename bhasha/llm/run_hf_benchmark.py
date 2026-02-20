import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

QUESTIONS = [
    {
        "category": "Translation",
        "prompt": "Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:\n\n'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'"
    },
    {
        "category": "Summarization",
        "prompt": "এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):\n\n'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'"
    },
    {
        "category": "OCR Fix",
        "prompt": "Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:\n\n'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)"
    },
    {
        "category": "Creative",
        "prompt": "একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।"
    }
]

MD_FILEPATH = "/home/benaaf/Desktop/BhashaLLM/llm outputs/bn_rag_8B.md"

def main():
    model_id = "BanglaLLM/bangla-llama-13b-base-v0.1"
    print(f"Loading {model_id} from Hugging Face...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    results_to_append = f"\n\n## Benchmark Outputs (HuggingFace Backend - {model_id})\n\n"

    for q in QUESTIONS:
        print(f"Running Category: {q['category']}")
        prompt_text = f"User: {q['prompt']}\nAssistant:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
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
        # Extract everything after Assistant:
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[1].strip()
        else:
            response = generated_text.replace(prompt_text, "").strip()
            
        time_taken = round(end_time - start_time, 2)
        results_to_append += f"### {q['category']}\n\n"
        results_to_append += f"**Prompt:**\n> {q['prompt'].replace(chr(10), chr(10) + '> ')}\n\n"
        results_to_append += f"**Response:** ({time_taken}s)\n\n{response}\n\n---\n"
        
        print(f"Completed {q['category']} in {time_taken}s")

    with open(MD_FILEPATH, "a", encoding="utf-8") as f:
         f.write(results_to_append)
         
    print(f"Results appended to {MD_FILEPATH}")

if __name__ == '__main__':
    main()
