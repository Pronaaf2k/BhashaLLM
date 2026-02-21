import os
import json
import pandas as pd
import random

def format_chatml(system, user, assistant):
    text = ""
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{assistant}<|im_end|>"
    return text

output_data = []

# 1. Spelling Checker (archive0)
try:
    with open('/home/benaaf/Desktop/BhashaLLM/datasets/extracted/archive0/right_file.txt', 'r', encoding='utf-8') as rf, \
         open('/home/benaaf/Desktop/BhashaLLM/datasets/extracted/archive0/wrong_file.txt', 'r', encoding='utf-8') as wf:
        rights = [line.strip() for line in rf][:20000] # Limit to 20K
        wrongs = [line.strip() for line in wf][:20000]
        for right, wrong in zip(rights, wrongs):
            if right and wrong:
                q = f"নিচের ভুল বানানের শব্দটি শুদ্ধ করে লিখুন: '{wrong}'"
                a = f"শুদ্ধ বানানটি হলো: '{right}'"
                output_data.append({"text": format_chatml("You are a helpful language expert.", q, a)})
except Exception as e:
    print(f"Error loading spelling data: {e}")

# 2. Morphology (archive2)
try:
    df_morph = pd.read_csv('/home/benaaf/Desktop/BhashaLLM/datasets/extracted/archive2/bangla_morphological_dataset.csv')
    for _, row in df_morph.iterrows():
        sentence = row['Data']
        label = row['Label']
        q = f"নিচের বাক্যটি আক্ষরিক (Literal) নাকি রূপক (Metaphorical) তা নির্ণয় করুন: '{sentence}'"
        a = f"বাক্যটি '{label}'।"
        output_data.append({"text": format_chatml("You are a helpful language expert.", q, a)})
except Exception as e:
    print(f"Error loading morphology data: {e}")

# 3. 80k Words (archive1) - generate some vocab queries
try:
    df_words = pd.read_csv('/home/benaaf/Desktop/BhashaLLM/datasets/extracted/archive1/bangla_word_80K_dataset.csv')
    words = df_words.iloc[:, 0].dropna().tolist()
    random.shuffle(words)
    words = words[:10000] # 10K words
    for word in words:
        word = str(word).strip()
        if word:
            q = f"এই শব্দটি কি একটি সঠিক বাংলা শব্দ? '{word}'"
            a = f"হ্যাঁ, '{word}' একটি সঠিক বাংলা শব্দ।"
            output_data.append({"text": format_chatml("You are a helpful language expert.", q, a)})
except Exception as e:
    print(f"Error loading words data: {e}")

# 4. OSCAR Sentences (archive3) - sample sentences for proofreading/grammar tasks
try:
    with open('/home/benaaf/Desktop/BhashaLLM/datasets/extracted/archive3/dataset.txt', 'r', encoding='utf-8') as f:
        sentences = []
        for i in range(50000): # read first 50K instead of all 1.5GB
            line = f.readline()
            if not line: break
            s = line.strip()
            if len(s) > 10:
                sentences.append(s)
        
        for s in sentences[:10000]: # Add 10K grammatical proofreading
            q = f"নিচের বাক্যটিতে কোনো ভাষার ভুল থাকলে সংশোধন করুন: '{s}'"
            a = f"বাক্যটি সঠিক আছে: '{s}'"
            output_data.append({"text": format_chatml("You are a helpful language expert.", q, a)})
except Exception as e:
    print(f"Error loading oscar data: {e}")

random.shuffle(output_data)

out_file = '/home/benaaf/Desktop/BhashaLLM/data/processed/kaggle_instruct.jsonl'
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, 'w', encoding='utf-8') as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Successfully generated {len(output_data)} training examples at {out_file}")
