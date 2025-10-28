from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

ocr_text = "আমার নাম চকলদার। আমি একজ জন ছাত্র।"
prompt = f"শুদ্ধ বাংলা বানান এবং ব্যাকরণে ঠিক করুন: {ocr_text}"

result = corrector(prompt, max_new_tokens=100)

print("🔤 Original OCR Text:")
print(ocr_text)
print("\n✅ Corrected Text:")
print(result[0]['generated_text'])
