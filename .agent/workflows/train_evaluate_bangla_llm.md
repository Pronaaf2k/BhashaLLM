---
description: How to train, pick best models, and evaluate the Bangla LLM
---

This workflow outlines the systematic process to train the Bangla LLM, extract the best performing checkpoints, and benchmark the model against specific language tasks.

1. **Train the Model**
   Run the training script to fine-tune the model using the prepared `kaggle_instruct.jsonl` instruction dataset. This will periodically save checkpoints into the `models/instruct_adapters` directory.
   // turbo
   ```bash
   python train_instruct.py
   ```

2. **Pick Best Checkpoints**
   After training concludes or is interrupted, use the selection script. It parses `trainer_state.json`, ranks the saved checkpoints by lowest `eval_loss`, and copies the top models into the `TRAINED MODELS` directory.
   // turbo
   ```bash
   python pick_best_models.py
   ```

3. **Run Benchmark Evaluation**
   Test the fine-tuned adapter (e.g., `final_instruct_adapter`) against predefined tasks such as spelling correction, grammar checking, translation, and summarization. The script outputs results to `llm outputs/qwen_bangla.md`.
   // turbo
   ```bash
   python bhasha/llm/run_qwen_bangla_benchmark.py
   ```

4. **Manual Review and Documentation**
   Open the generated markdown output (e.g., `llm outputs/qwen_bangla.md`) and manually review the model's generated responses. Update the table grades (Passed/Missed/Hallucinated) and final verdicts based on the model's performance on the specific task.
