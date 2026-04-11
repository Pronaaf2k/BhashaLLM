# BhashaLLM Directory Structure

This file provides a map of the repository to guide codebase navigation and future refactoring efforts.

## Root Directories

- `bhasha/`: The core Python package for the project. All active application code belongs here.
  - `bhasha/app/`: Core application logic or UI code.
  - `bhasha/data/`: Data loading, processing, and pipeline logic.
  - `bhasha/eval/`: Evaluation logic and metrics calculations.
  - `bhasha/llm/`: Large Language Model training, generation, and wrapper code.
  - `bhasha/ocr/`: Optical Character Recognition model code and utilities.
  - `bhasha/scripts/`: Standalone helper and utility scripts (e.g., downloading models, preparing datasets, running specific training configurations).
  - `bhasha/utils/`: Shared helper functions across the codebase.
- `docs/`: Markdown files documenting project plans, APIs, and summaries.
- `tests/`: Automated tests, structured to match the `bhasha/` hierarchy.
- `logs/`: Application logs, training logs, and error outputs.
- `datasets/`: Scripts and metadata for handling and preparing models' datasets.
- `data/`: Raw and processed dataset files (often `.json`, `.csv`, `.jsonl`, or images).
- `models/`: Checkpoints and artifacts for base models and trained models.
- `trainedmodels/`: Checkpoints and weights for fine-tuned models.
- `report/`: Output logs, charts, and final markdown reports generated during evaluation.
- `llm outputs/`: Output responses and generations from various language models for benchmarks.

## Root Files

- `main.py`: The single entry point to execute the CLI or core application logic.
- `setup.py` / `requirements.txt` / `environment.yml`: Dependency and package management files.
- `README.md`: High-level entry overview of the project.

## Note for AI Assistants
When performing refactors:
1. Always keep utility and one-off scripts within `bhasha/scripts/` to avoid polluting the root directory.
2. Put all tests inside `tests/` instead of alongside application code.
3. Keep all documentation in `docs/` (except the main `README.md`).
4. Avoid placing `.log`, `.txt` outputs, or loose `.py` files in the root folder.
