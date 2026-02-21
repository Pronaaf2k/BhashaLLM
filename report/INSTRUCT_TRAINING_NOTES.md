# Instruction Tuning Phase: Training Notes

This document contains a record of the parameters, datasets, and pipeline configurations used to fine-tune `Qwen-2.5-1.5B-Instruct` on several Kaggle datasets for better downstream Bangla performance. This information can be directly integrated into the final research report.

## 1. Datasets & Preparation

The raw data was initially downloaded manually from Kaggle into the `datasets/` folder. The following datasets were combined into an instruction-following format (mapping queries and answers into Qwen's ChatML prompt format `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...`):

1. **Spelling Checker v1**: Generated ~20,000 spelling correction tasks (*"Correct the spelling of this word..."*).
2. **Bangla Morphological Dataset**: Generated literal vs. metaphorical classification tasks.
3. **Bangla Word Frequency 80k**: Sampled 10,000 vocabulary queries indicating if a word is natively valid or not.
4. **Bengali Sentences from OSCAR**: Sampled 10,000 sentences for basic proofreading/grammatical integrity queries.

**Total Processed Dataset Size**: `43,258` examples.  
The dataset was randomly split internally using `datasets` into:
- **Train Split**: 38,932 examples
- **Evaluation Split**: 4,326 examples

The output was saved to `data/processed/kaggle_instruct.jsonl`.

## 1.5 The Evaluation "Quiz" Strategy

To ensure the model learns rather than just memorizes the training data, a strict evaluation block was implemented during training:

*   **The Holdout Set:** Initially, **4,326 examples** (10% of the data) were surgically extracted into an isolated `eval_dataset`. The model is *never* allowed to update its weights based on these examples during the training loops.
*   **The "Pop Quizzes"**: Every 20 training steps (`eval_steps=20`), the model is paused from learning and forced to iterate over those 4,326 examples, computing its `eval_loss`.
*   **Significance**: These "quizzes" test the model's actual general-purpose understanding of the Bangla instructions (e.g. Can it correct the spelling of a word it has never seen?). The lower this validation loss goes, the more confident we can be that the LoRA adapter is successfully learning robust conversational boundaries and mapping real grammar rules, rather than just overfitting.

### Example Evaluation Quiz Questions:

During the quiz, the model is fed exactly the `System` and `User` portions of unseen training rows, and the trainer algorithm mathematically compares its response against the hidden, correct `Assistant` response.

**Example 1: Spelling Error Check**
> **System:** You are a helpful language expert.
> **User:** নিচের ভুল বানানের শব্দটি শুদ্ধ করে লিখুন: 'দূরারোগ্য'
> 
> *Expected Assistant Answer:* শুদ্ধ বানানটি হলো: 'দুরারোগ্য'

**Example 2: Valid Vocabulary check**
> **System:** You are a helpful language expert.
> **User:** এই শব্দটি কি একটি সঠিক বাংলা শব্দ? 'হনূমান'
> 
> *Expected Assistant Answer:* না, 'হনূমান' একটি সঠিক বাংলা শব্দ নয়। সঠিক শব্দটি হলো 'হনুমান'।

## 2. Base Model & Precision

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization (BitsAndBytes)**: 
  - 4-bit loading enabled (`load_in_4bit=True`).
  - Quantization type: `nf4`.
  - Nested Quantization (`bnb_4bit_use_double_quant=True`).
  - Compute Dtype: `torch.float16`.

## 3. PEFT (LoRA) Parameters

Efficient fine-tuning was employed using LoRA targeting all major logical projections in the transformer layers:
- **Rank (`r`)**: 16
- **Alpha (`lora_alpha`)**: 32
- **Dropout**: 0.05
- **Task Type**: `CAUSAL_LM`
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## 4. Hyperparameters & Training Setup

The training was run via TRL's `SFTTrainer` with the following hyperparameters configuration passed via CLI (`args`) and internal defaults:

- **Per-Device Train Batch Size**: 1
- **Gradient Accumulation Steps**: 4
- **Learning Rate**: 2e-4
- **Epochs**: 1
- **Max Context Length**: 512
- **Mixed Precision**: bf16 (`bf16=True`)
- **Optimizer**: `paged_adamw_32bit`
- **Evaluation/Saving Strategy**: Evaluated and checkpoints saved every 20 steps.

## summary of process commands
To reproduce this run:
1. `venv/bin/python bhasha/llm/prepare_kaggle_data.py` - Builds the structured instruction set.
2. `venv/bin/python bhasha/llm/train_instruct.py --base_model Qwen/Qwen2.5-1.5B-Instruct --batch_size 1 --epochs 1 --context_length 512` - Runs the tuner.
