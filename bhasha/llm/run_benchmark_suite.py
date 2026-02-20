import requests
import json
import time
import os

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "Qwen_1.5B": "qwen2.5:1.5b",
    "Gemma_2B": "gemma2:2b",
    "Qwen_3B": "qwen2.5:3b",
    "Mistral_7B": "mistral:latest",
    "Gemma_9B": "gemma2:9b",
    "Nemo_12B": "mistral-nemo:latest",
    "Llama_3.1_8B": "llama3.1:latest",
    "Llama_3.2_11B": "llama3.2:latest",
    "bn_rag_8B": "hf.co/BanglaLLM/bangla-llama-13b-base-v0.1-GGUF"
}

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

OUTPUT_DIR = "/home/benaaf/Desktop/BhashaLLM/llm outputs"

def generate_response(model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3, # using low temp for consistent tests
            "top_p": 0.9,
            "num_predict": 256
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"Error: Model '{model_name}' not found locally. Please pull the model using `ollama pull {model_name}`"
        return f"Error: {e}"
    except Exception as e:
        print(f"Error calling Ollama API for model {model_name}: {e}")
        return f"Error: {e}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for md_filename, ollama_model in MODELS.items():
        md_filepath = os.path.join(OUTPUT_DIR, f"{md_filename}.md")
        
        print(f"==================================================")
        print(f"Running Benchmark for: {md_filename} ({ollama_model})")
        print(f"==================================================")
        
        results_to_append = f"\n\n## Benchmark Outputs (Ollama Backend)\n\n"
        
        for q in QUESTIONS:
            print(f"Category: {q['category']}")
            start_time = time.time()
            response_text = generate_response(ollama_model, q['prompt'])
            end_time = time.time()
            
            time_taken = round(end_time - start_time, 2)
            results_to_append += f"### {q['category']}\n\n"
            results_to_append += f"**Prompt:**\n> {q['prompt'].replace(chr(10), chr(10) + '> ')}\n\n"
            results_to_append += f"**Response:** ({time_taken}s)\n\n{response_text}\n\n---\n"
            
            print(f"Completed in {time_taken}s")
            print("---")
            
        print(f"Writing outputs to {md_filepath}")
        try:
            with open(md_filepath, "a", encoding="utf-8") as f:
                f.write(results_to_append)
            print(f"Successfully appended results to {md_filepath}")
        except IOError as e:
            print(f"Failed to write to {md_filepath}: {e}")
            
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
