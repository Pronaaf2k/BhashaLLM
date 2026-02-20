# Unsloth Fine-Tuning Notes: Optimizing Qwen 1.5B for Low-End Devices

Based on the [Unsloth Documentation](https://unsloth.ai/docs), here is a comprehensive breakdown of why and how we will use Unsloth to fine-tune our Qwen model for the `BhashaLLM` project on our 16GB VRAM hardware.

## 🦥 What is Unsloth?
Unsloth is an open-source framework specifically optimized for fine-tuning Large Language Models (LLMs) significantly faster and highly memory-efficient compared to standard Hugging Face Transformers.
* **Performance:** It offers **2x faster training** speeds and uses **70% less VRAM** with **0% loss in accuracy**.
* **Methodology:** It utilizes highly optimized custom Triton kernels and backends instead of relying heavily on approximations. It fully supports exact calculation implementations for LoRA and QLoRA.
* **Ecosystem Compatibility:** It seamlessly exports fine-tuned models to Ollama, llama.cpp (GGUF), and vLLM. 

## ⚙️ Key Advantages for Tuning Qwen 1.5B
Since our device has limited VRAM (16GB), Unsloth provides several game-changing benefits:

1. **4-bit Quantization First:** By enabling `load_in_4bit = True`, Unsloth reduces the memory footprint of the model by 4x. This allows us to load the Qwen 1.5B base/instruct model and the optimizer states easily without hitting OOM (Out of Memory) errors.
2. **Architecture Bug Fixes:** The Unsloth team directly collaborates with the Qwen team. They have fixed critical bugs inside Qwen's code (specifically around dynamic GGUF exports and 128k context handling) that improve the accuracy of our downstream tuning.
3. **8x Longer Context Fine-Tuning:** Unsloth uses YaRN and RoPE scaling efficiently. Although Qwen supports up to 40k+ context naturally, for a low-end GPU, Unsloth recommends keeping `max_seq_length = 2048` for initial testing. Memory overhead scales quadratically with context length, so keeping it short avoids VRAM spikes.
4. **Mixed Datasets for Reasoning:** If we decide to use the reasoning variants of Qwen, Unsloth documentation suggests mixing datasets to preserve intelligence (e.g., maintaining a ratio like 75% reasoning data and 25% conversational/non-reasoning data).

## 🛠️ Typical Implementation Flow for Qwen in Unsloth
When we write the training script, the flow will look like this:

1. **Initialize the Model & Tokenizer using `FastLanguageModel`:**
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "Qwen/Qwen2.5-1.5B-Instruct",
       max_seq_length = 2048,
       dtype = None, # Auto detects fp16/bf16
       load_in_4bit = True # Critical for low VRAM
   )
   ```

2. **PEFT (Parameter-Efficient Fine-Tuning) Setup:**
   Attach LoRA adapters to specific target modules (like `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Unsloth automatically applies its fast Triton kernels to these modules.

3. **Format the Dataset:**
   Map the BhashaLLM OCR correction/instruction data into the standard ChatML or Qwen format. Unsloth provides standard mappers for this.

4. **Train using SFTTrainer:**
   Use the standard Hugging Face `SFTTrainer` from the `trl` library, passing in the Unsloth-optimized model. We will use `paged_adamw_8bit` as the optimizer to save even more VRAM.

5. **Direct GGUF Export:**
   Once trained, Unsloth has one-click export functions: `model.save_pretrained_gguf("model_name", tokenizer, quantization_method="q4_k_m")` which perfectly aligns with our Ollama deployment pipeline.
