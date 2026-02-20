# Gemma 9B

| Metric | Result |
| :--- | :--- |
| Translation | ⚠️ Hindi Chars |
| Summarization | ⭐ Bangla |
| OCR Fix | ✅ Perfect |
| Creative | ⭐ Good |
| Verdict | **Script Issues** |


## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate the following English paragraph into natural, standard Bangla. Ensure the tone is polite and professional:
> 
> 'The rapid advancement of Artificial Intelligence is reshaping how we approach software development. However, ensuring data privacy remains a significant challenge for startups in the current ecosystem.'

**Response:** (6.03s)

"कृत्रিম বুদ্ধিমত্তার দ্রুতগতির উন্নয়ন সফটওয়্যার উন্নয়নের পদ্ধতি পরিবর্তন করছে। তবে, বর্তমানে শুরুতে প্রতিষ্ঠানগুলির জন্য ডেটা গোপনীয়তা নিশ্চিত করা একটি বিরাট চ্যালেঞ্জ হয়ে উঠেছে।"


This translation maintains a polite and professional tone while accurately conveying the meaning of the original English paragraph. 

Here's a breakdown:

* **कृत्रিম বুদ্ধিমত্তার দ্রুতগতির উন্নয়ন (Krtrīm buddhimāttar drutgatira unnaya)** -  "Rapid advancement of Artificial Intelligence"
* **সফটওয়্যার উন্নয়নের পদ্ধতি পরিবর্তন করছে (Softwār unnāyoner poddhati paribartana koreche)** - "is reshaping how we approach software development"
* **তবে, (Tbe,)** - "However,"
* **বর্তমানে শুরুতে প্রতিষ্ঠানগুলির জন্য (Bartamanē shuru te pratiṣṭhānigulirer jonye)** - "for startups in the current ecosystem"
* **ডেটা গোপনীয়তা নিশ্চিত করা একটি বিরাট চ্যালেঞ্জ হয়ে উঠেছে (Dāta goponiyāta nishchhita kora ekti biraṭ chālēñja hōye utheche)** - "ensuring data privacy remains a significant challenge"



Let me know if you have any other text you'd like me to translate!

---
### Summarization

**Prompt:**
> এই অনুচ্ছেদটি একটি বাক্যে সারসংক্ষেপ করো (Summarize in one sentence):
> 
> 'বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা। বর্তমানে কৃত্রিম বুদ্ধিমত্তার যুগে বাংলার সঠিক ব্যবহার নিশ্চিত করা আমাদের জন্য বড় চ্যালেঞ্জ। বিশেষ করে হাতে লেখা বাংলা অক্ষর ডিজিটাল ফরম্যাটে রূপান্তর করার ক্ষেত্রে অনেক সীমাবদ্ধতা রয়েছে যা আধুনিক এলএলএম (LLM) ব্যবহারের মাধ্যমে সমাধান করা সম্ভব।'

**Response:** (0.78s)

বাংলা ভাষার ডিজিটাল ফরম্যাটে রূপান্তর, বিশেষ করে হাতে লেখা অক্ষর, আধুনিক AI প্রযুক্তির সাহায্যে সম্ভব।

---
### OCR Fix

**Prompt:**
> Fix the spelling and grammatical errors in this broken OCR output. Do not add extra commentary:
> 
> 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)

**Response:** (0.49s)

আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।

---
### Creative

**Prompt:**
> একটি ছোট গল্প লিখুন (৫টি বাক্য) যেখানে একটি রোবট এবং একটি ছোট ছেলে বৃষ্টির দিনে চা খাচ্ছে।

**Response:** (2.25s)

বৃষ্টি জমজমা করে পড়ছিল, আকাশ ধুলায় ভাসিয়ে ফেলেছিল।  সামান্থের বাগানের ছোট্ট একটি উইন্ডোতে বসে ছোট ছেলে তনভি চা খাচ্ছিল। তার সঙ্গে ছিল তার রোবট বন্ধু, জ্যাক, যার হাতে থাকা গরম চা কাপটি তনভির কাছে পৌঁছে দিয়েছিল।  তনভি জ্যাকের ধাতব মুখ দেখল এবং বলল, "এটা বৃষ্টির দিনে খুব ভালো লাগে।" জ্যাক স্মরণীয়ভাবে কথা বলল, "আমিও তোমার সাথে চা পান করতে ভালোবাসি, তনভি।"

---
