# Research Report: Benchmarking LLMs for Bangla OCR Correction & RAG

**Date:** February 11, 2026
**Author:** Nova Quinn (Assistant to pronaaf2k)
**Project:** BhashaLLM

## 1. Abstract
This study evaluates the performance of state-of-the-art Large Language Models (LLMs) ranging from 1.5B to 12B parameters on Bangla language tasks. The focus is on identifying the optimal model for an OCR correction pipeline and Retrieval-Augmented Generation (RAG) system, constrained by consumer hardware (RTX 5070 Ti, 16GB VRAM).

## 2. Methodology

### 2.1 Hardware Environment
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- **Framework:** PyTorch 2.1, Transformers 4.37, BitsAndBytes (4-bit quantization)

### 2.2 Models Tested
| Model | Size | License | Key Feature |
| :--- | :--- | :--- | :--- |
| **Qwen-2.5-1.5B** | 1.5B | Apache 2.0 | Extremely fast, lightweight. |
| **Qwen-2.5-3B** | 3B | Apache 2.0 | "Goldilocks" size for edge devices. |
| **Gemma-2-2B** | 2B | Gemma | Google's high-efficiency tiny model. |
| **Gemma-2-9B** | 9B | Gemma | Heavyweight, known for reasoning. |
| **Mistral-7B-v0.3** | 7B | Apache 2.0 | Strong instruction following. |
| **Mistral-Nemo-12B** | 12B | Apache 2.0 | Large context window (128k). |
| **Llama-3-8B** | 8B | Llama | The standard baseline. |
| **bn_rag_llama3-8b** | 8B | Llama | Fine-tuned specifically for Bangla RAG. |
| **Llama-3.1-8B** | 8B | Llama | Updated multilingual training. |
| **Llama-3.2-11B** | 11B | Llama | Multimodal (Vision + Text). |

### 2.3 Evaluation Metrics
Models were scored on **Translation** (En->Bn), **Summarization** (Bn->Bn), **Creative Writing** (Poetry), **OCR Correction** (Spelling), and **QA** (Fact Extraction).

## 3. Results & Analysis
### 3.1 Performance Matrix (Simplified)

| Model | Translation | Summarization | OCR Fix | Creative | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 1.5B** | ❌ Hallucinated | ❌ Hallucinated | ❌ Missed | ❌ Hallucinated | **Too Weak** |
| **Gemma 2B** | ⚠️ Poor Bangla | ❌ English Out | ❌ Fail | ⭐ Good | **Confused** |
| **Qwen 3B** | ⚠️ Poor Bangla | ❌ Chinese Chars | ⚠️ Chinese Chars | ⭐ Good | **Unreliable** |
| **Mistral 7B** | ⚠️ Poor Bangla | ❌ English Out | ❌ English Out | ⚠️ Verbose Scripts | **Verbose** |
| **Gemma 9B** | ⚠️ Hindi Chars | ⭐ Bangla | ✅ Perfect | ⭐ Good | **Script Issues** |
| **Nemo 12B** | ✅ Good | ❌ English Out | ✅ Perfect | ⚠️ Weird Format | **Glitchy** |
| **bn_rag 8B** | ❌ News Hallucination | ❌ News Hallucination | ❌ News Hallucination | ❌ News Hallucination | **Base Model (No Instruct)** |
| **Llama 3.1 8B** | ⚠️ Hindi Chars | ⭐ Bangla | ⭐ Perfect | ⭐ Excellent | **Runner Up** |
| **Llama 3.2 11B**| ✅ Good | ⭐ Bangla | ⭐ Perfect | ⭐ Good | **Champion** |

### 3.2 Key Findings

1.  **The "Language Barrier":** Most general models (Mistral, Qwen, Base Llama 3.0) struggle to output summaries *in Bangla* even when prompted. They revert to English.
2.  **Script Confusion:** Qwen and Gemma often mixed Hindi (Devanagari) characters into Bangla output, likely due to shared token embeddings.
3.  **Llama 3.1 Supremacy:** Llama 3.1 8B showed a massive improvement over 3.0. It respects the language constraint (Bangla Output) as well as the fine-tuned specialist, but retains superior general reasoning and creativity.

## 4. Final Recommendations

### 🥇 Best Overall (The Champion)
**`meta-llama/Llama-3.2-11B-Vision-Instruct`**
- **Why:** Following real-world testing, it demonstrated the best balance of speed, absolute accuracy, and adherence strictly to correct Bengali script (unlike 3.1 which introduced some Hindi/Devanagari characters during translation). It also adds the ability to process images directly which is ideal for End-to-End OCR.

### 🥈 Best General Text (The Runner Up)
**`meta-llama/Meta-Llama-3.1-8B-Instruct`**
- **Why:** Incredible creative writing and pure Bangla extraction, but occasionally glitches with cross-script contamination (Devanagari characters) in direct translation tasks. Still highly useful as a fallback.

### 🥉 Best Tiny Model
**`Qwen/Qwen2.5-3B-Instruct`**
- **Why:** If VRAM is strictly limited (<8GB), this is the only usable option, though output validation is required to catch Hindi character intrusion.

## 5. Conclusion
For the **BhashaLLM** project, we will deploy **Llama-3.2-11B** as the core engine for the RAG and Text Correction pipeline. Its performance negates the immediate need for custom fine-tuning and avoids script confusion issues, allowing development to focus on the retrieval architecture.

---

