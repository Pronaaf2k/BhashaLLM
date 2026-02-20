# Gemma 2B

| Metric | Result |
| :--- | :--- |
| Translation | ⚠️ Poor Bangla |
| Summarization | ❌ English Out |
| OCR Fix | ❌ Fail |
| Creative | ⭐ Good |
| Verdict | **Confused** |


## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:
> 
> 'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'

**Response:** (2.57s)

"আন্তর্জাতিক আইনীয়তা উন্নয়নের দ্রুত প্রবাহ মানসম্পর্কিত সফটওয়্যার ডেভেলপমেন্টের নীতির পরিবর্তন করে। কিন্তু, মূল্যবান তথ্য ব্যক্তিগততা রক্ষা করা বেশিরভাগ শুরুআইনের জন্য একটি গুরুত্বপূর্ণ প্রশ্ন।" 


**Explanation of the translation:**

* **আন্তর্জাতিক আইনীয়তা উন্নয়নের দ্রুত প্রবাহ**: This translates to "the rapid advancement of Artificial Intelligence is reshaping how we approach software development."
* **মানসম্পর্কিত সফটওয়্যার ডেভেলপমেন্টের নীতির পরিবর্তন**:  This translates to "software development's methodology is changing" 
* **কিন্তু, মূল্যবান তথ্য ব্যক্তিগততা রক্ষা করা বেশিরভাগ শুরুআইনের জন্য একটি গুরুত্বপূর্ণ প্রশ্ন**: This translates to "However, ensuring data privacy remains a significant challenge for startups in the current ecosystem." 


**Note:**

* The translation uses formal language and avoids slang or colloquial terms.
* It maintains a polite and professional tone throughout the paragraph.

---
### Summarization

**Prompt:**
> এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):
> 
> 'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'

**Response:** (0.28s)

The Bengali language, rich in its own right, presents a challenge in the digital age as translating traditional writing into digital formats remains limited, but modern LLMs offer potential solutions.

---
### OCR Fix

**Prompt:**
> Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:
> 
> 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

**Response:** (0.45s)

'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

---
### Creative

**Prompt:**
> একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।

**Response:** (0.58s)

আকাশের উপরে বিকেলের ধারণা গুলি করে, একটি স্পষ্ট কালো রঙের রোবট জিজ্ঞাসা করল: "তুমি কি চা খাবো?"  এই ছোট ছেলেটির হাতে একটি পাকা বৃষ্টিপাতের উপরে ছিল।  "আমার জন্য," তিনি বললেন, "সবাই ভালো থাকে!"

---
