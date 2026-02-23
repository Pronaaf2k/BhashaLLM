# Qwen Bangla (Qwen2.5-1.5B-Instruct + Fine-tuned Adapter)

| Metric | Result |
| :--- | :--- |
| Spelling Correction | ✅ Passed |
| Grammar Checking | ❌ Missed |
| Word Validation | ✅ Passed |
| Metaphor vs Literal | ✅ Passed |
| Verdict | **Highly Specialized** |


## Benchmark Outputs

### Spelling Correction

**Prompt:**
> নিচের ভুল বানানের শব্দটি শুদ্ধ করে লিখুন: 'দূরারোগ্য'

**Response:** (1.89s)

শুদ্ধ বানানটি হলো: 'দুরারোগ্য'

---
### Grammar Checking

**Prompt:**
> নিচের বাক্যটিতে কোনো ভাষার ভুল থাকলে সংশোধন করুন: 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।'

**Response:** (3.29s)

বাক্যটি সঠিক আছে: 'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।'

---
### Word Validation

**Prompt:**
> এই শব্দটি কি একটি সঠিক বাংলা শব্দ? 'রোবট'

**Response:** (1.55s)

হ্যাঁ, 'রোবট' একটি সঠিক বাংলা শব্দ।

---
### Metaphor vs Literal

**Prompt:**
> নিচের বাক্যটি আক্ষরিক (Literal) নাকি রূপক (Metaphorical) তা নির্ণয় করুন: 'অতিভক্তি চোরের লক্ষণ'

**Response:** (0.57s)

বাক্যটি 'Metaphor'।

---
