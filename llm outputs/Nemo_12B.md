# Nemo 12B

| Metric | Result |
| :--- | :--- |
| Translation | ✅ Good |
| Summarization | ❌ English Out |
| OCR Fix | ✅ Perfect |
| Creative | ⚠️ Weird Format |
| Verdict | **Glitchy** |


## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:
> 
> 'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'

**Response:** (10.27s)

"কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence) এর দ্রুত অগ্রগতি সফটওয়্যার উন্নয়নের উপায়কে পরিবর্তন করছে। কিন্তু, বর্তমান পরিবেশে স্টার্টআপদের ডেটা গোপনীয়তা নিশ্চিত করা একটি উল্লেখযোগ্য চ্যালেঞ্জ রয়ে গেছে।"

---
### Summarization

**Prompt:**
> এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):
> 
> 'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'

**Response:** (0.61s)

"Bengali, a rich global language, faces challenges in accurate usage during the era of artificial intelligence, with many hurdles in converting handwritten Bengali to digital formats, which modern LLM models can help overcome."

---
### OCR Fix

**Prompt:**
> Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:
> 
> 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

**Response:** (0.46s)

Here's the corrected version:

আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।

---
### Creative

**Prompt:**
> একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।

**Response:** (1.43s)

একটি রোবট এবং একটি ছোট ছেলে, তারা চা পেয়েছে
**সকালের আতশ্বজ্যিতে তারা বসেছিলো**
**ছেলেটি চায়ের তালি হাতে ধরে ছিলো**
**রোবটটি তার মেকানিক্যাল হাত দিয়ে চা ঘুমানো করছিলো**
**দুইজনেই শান্তভাবে চা পেয়ে খাচ্ছিলো**
**একটি সুন্দর স্মৃতি রেখে গিয়েছিলো তাদের মধ্যে।**

---
