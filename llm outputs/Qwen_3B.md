# Qwen 3B

| Metric | Result |
| :--- | :--- |
| Translation | ⚠️ Poor Bangla |
| Summarization | ❌ Chinese Chars |
| OCR Fix | ⚠️ Chinese Chars |
| Creative | ⭐ Good |
| Verdict | **Unreliable** |


## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:
> 
> 'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'

**Response:** (5.42s)

আমাদের সফটওয়্যার প্রোগ্রামিং নেতৃত্বে অ্যারিট্যাকল ইইন্ফরমেশন ইকৌচের গড়ে উঠা প্রকৌশাল তথাকথিত AI, বিশ্বব্যাংকরতা অপেক্ষাকৃতভাবে মজাদাবাদ করছে। আমরা আধুনিক ইকৌচের স্টারটআপ প্রোগ্রামিং নেতৃত্বে ডেটা পরিষ্কারতা নির্ধারণ করতে যথেষ্ট সমস্তচর্চা হচ্ছে।

---
### Summarization

**Prompt:**
> এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):
> 
> 'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'

**Response:** (0.75s)

বাংলা বিশ্বের অন্যতম সমৃদ্ধ ভাষা, তব然 ডিজিটাল ফরম্যাটে হাতে লেখা বাংলা অক্ষরের রূপান্তর করার সমস্যাগুলি আধুনিক LLM এর মাধ্যমে সমাধান করা প্রতিষ্ঠিতভাবে উপযোগী।

---
### OCR Fix

**Prompt:**
> Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:
> 
> 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

**Response:** (0.34s)

আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই。

---
### Creative

**Prompt:**
> একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।

**Response:** (1.77s)

এক বৃষ্টির দিনে, রোবটটি একটি ছোট ছেলের সাথে গল্প করতে যায়। তারা বৃষ্টির ভিতর দিয়ে নদীতে এগিয়ে যায়, তারপর ছোট ছেলে স্কুলে ফিরে আসা পথে বৃষ্টিশীল গাছগুলিতে চা খায়। রোবটটি ছোট ছেলেকে সহযোগিতায় বৃষ্টির ভিতর দিয়ে গাছপালা থেকে চা নিতে প্রচেşতে। এই সব ঘটনার মধ্যে, বৃষ্টির ভিতরে কয়েকটি ছোট গাথা ছোট ছেলেকে আহত করেছিল। সমস্যার মধ্যে ছোট ছেলে এবং রোবট তাদের প্রচেşনা নির্বাহী করতে সমস্ত ঘটনাটি কেনা করে।

---
