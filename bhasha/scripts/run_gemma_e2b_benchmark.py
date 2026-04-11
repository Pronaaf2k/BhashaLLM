import requests
import time
import os

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "gemma-4-E2B-it"

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
MD_FILEPATH = os.path.join(OUTPUT_DIR, "Gemma_4_E2B.md")

def generate_response(prompt):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.9,
        # Allow enough tokens for Chain-of-Thought reasoning + actual response
        "max_tokens": 1024 
    }
    try:
        response = requests.post(LLAMA_SERVER_URL, json=payload)
        response.raise_for_status()
        
        msg = response.json()["choices"][0]["message"]
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning_content", "").strip()
        
        # Format the output to show reasoning if present
        final_output = ""
        if reasoning:
           final_output += f"<details><summary>Thought Process</summary>\n\n```text\n{reasoning}\n```\n\n</details>\n\n"
        final_output += content if content else "(No final response generated, perhaps cut off?)"
        
        return final_output
    except Exception as e:
        print(f"Error calling llama-server API: {e}")
        return f"Error: {e}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"==================================================")
    print(f"Running Benchmark for: {MODEL_NAME}")
    print(f"==================================================")
    
    results_to_append = f"## Benchmark Outputs ({MODEL_NAME})\n\n"
    
    for q in QUESTIONS:
        print(f"Category: {q['category']}")
        start_time = time.time()
        response_text = generate_response(q['prompt'])
        end_time = time.time()
        
        time_taken = round(end_time - start_time, 2)
        results_to_append += f"### {q['category']}\n\n"
        results_to_append += f"**Prompt:**\n> {q['prompt'].replace(chr(10), chr(10) + '> ')}\n\n"
        results_to_append += f"**Response:** ({time_taken}s)\n\n{response_text}\n\n---\n"
        
        print(f"Completed in {time_taken}s")
        print("---")
        
    print(f"Writing outputs to {MD_FILEPATH}")
    try:
        with open(MD_FILEPATH, "w", encoding="utf-8") as f:
            f.write(results_to_append)
        print(f"Successfully wrote results to {MD_FILEPATH}")
    except IOError as e:
        print(f"Failed to write to {MD_FILEPATH}: {e}")
        
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
