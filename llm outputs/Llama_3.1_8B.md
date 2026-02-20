# Llama 3.1 8B

| Metric | Result |
| :--- | :--- |
| Translation | ⚠️ Hindi Chars |
| Summarization | ⭐ Bangla |
| OCR Fix | ⭐ Perfect |
| Creative | ⭐ Excellent |
| Verdict | **Runner Up** |


## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:
> 
> 'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'

**Response:** (7.88s)

আর্টিফিশियल ইন্টেলিজেন্স-এর দ্রুত অগ্রগতি কোডিংয়ের পদ্ধতিকে বহুমাত্রিক করছে। সফ্টওয়্যার উন্নয়নে আগ্রহী শুরুপাতা এবং স্টার্টআপদের জন্য ডেটা গোপনীয়তা নিশ্চিত করা এখনও একটি বড় চ্যালেঞ্জ।

---
### Summarization

**Prompt:**
> এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):
> 
> 'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'

**Response:** (1.17s)

বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার সময় আধুনিক LLM (Large Language Model) ব্যবহার করা এই অঞ্চলের জন্য উপযোগী।

---
### OCR Fix

**Prompt:**
> Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:
> 
> 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

**Response:** (0.65s)

আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।

---
### Creative

**Prompt:**
> একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।

**Response:** (2.47s)

বৃষ্টির দিনে, একজন ছোট শিশু তার মা-বাপের সাথে ঘরের আঙ্গিনায় চা খাচ্ছিল। একদিকে, একটি রোবট তাদের সাথে বসে আছে। শিশুটি রোবটটিকে চা পান করতে বলল, কিন্তু রোবটটি জানত যে এটি শুধুমাত্র চা খাবে না। সে পরিষ্কার পানি দিয়ে হাত ধোয়েছে, এবং শিশুটি খুশি হয়ে চা পান করল।

---
