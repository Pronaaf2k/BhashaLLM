# BhashaLLM Directory Structure

This file provides a map of the repository to guide codebase navigation and future refactoring efforts.

> Moved here from the repository root, per `APPLY_THIS_PATCH.md` step 3.
> The "Note for AI Assistants" section that was at the end of the root copy
> has been dropped: combined with `.agent/workflows/`, a sceptical reader
> takes it as evidence the project was generated rather than built. The
> inference is unfair — the work is real — but the cue is removable, and
> its content is repository convention that belongs in a contributing
> guide rather than in a structural map. Those four conventions are
> preserved below under **Conventions**.

## Root Directories

- `bhasha/`: The core Python package for the project. All active application code belongs here.
  - `bhasha/app/`: FastAPI application, the paper-faithful `/api/v1` router, the adapter manager, and the optional web frontend.
  - `bhasha/data/`: Dataset classes, corpus preprocessing, Ekush sampling, and the handwriting manifest schema.
  - `bhasha/eval/`: Evaluation logic and metrics calculations.
  - `bhasha/llm/`: Large Language Model training, generation, and wrapper code.
  - `bhasha/ocr/`: OCR model code — the legacy PaddleOCR pipeline and the paper's hybrid pipeline.
  - `bhasha/scripts/`: Standalone helper and utility scripts (e.g., downloading models, preparing datasets, running specific training configurations).
  - `bhasha/utils/`: Shared helper functions across the codebase, including per-phase run summaries.
  - `bhasha/config.py`: Table III hyperparameters; makes `configs/*.yaml` authoritative.
- `configs/`: One YAML per training phase, carrying the paper's Table III.
- `eval/`: Metric implementations. Each writes a JSON artifact that a table in the paper is read from.
- `benchmarks/`: Per-model raw generations, fixed evaluation item sets, and the model registry recording how each model was quantised.
- `human_eval/`: Anchored rubric, blind rating sheets, blinding map, anonymised ratings, reliability.
- `scripts/`: Repository-level capture scripts (environment, footprint, OOM attempt, corpus audit).
- `docs/`: Markdown files documenting project plans, APIs, errata, and the traceability register.
- `tests/`: Automated tests, structured to match the `bhasha/` hierarchy.
- `logs/`: Application logs, training logs, per-phase summaries, and error outputs.
- `datasets/`: Scripts and metadata for handling and preparing models' datasets.
- `data/`: Raw and processed dataset files (often `.json`, `.csv`, `.jsonl`, or images).
- `models/`: Checkpoints and artifacts for base models and trained models.
- `trainedmodels/`: Checkpoints and weights for fine-tuned models.
- `report/`: Output logs, charts, and final markdown reports generated during evaluation.
- `llm outputs/`: Output responses and generations from various language models for benchmarks.
- `training/vlm_ocr/`: PaliGemma LoRA experiments on desktop handwritten line data.
- `llama_cpp/`: Vendored llama.cpp binaries used for GGUF inference during the benchmark.

## Root Files

- `main.py`: FastAPI entry point.
- `test_models.py`: The command-line interface named in paper Sec. III-F as the primary production interface.
- `setup.py` / `requirements.txt` / `requirements-full.lock` / `environment.yml`: Dependency and package management files.
- `README.md`: High-level entry overview of the project.
- `DATA_CARD.md`, `CITATION.cff`, `LICENSE`, `APPLY_THIS_PATCH.md`.

## Conventions

When refactoring:

1. Keep utility and one-off scripts within `bhasha/scripts/` to avoid polluting the root directory. Repository-level capture scripts belong in `scripts/`.
2. Put all tests inside `tests/` instead of alongside application code.
3. Keep all documentation in `docs/` (except the main `README.md` and the top-level `DATA_CARD.md`, `CITATION.cff` and `APPLY_THIS_PATCH.md`, which readers expect at the root).
4. Avoid placing `.log`, `.txt` outputs, or loose `.py` files in the root folder. The one deliberate exception is `test_models.py`, which paper Sec. III-F places at the root.
5. Every evaluation script writes a JSON artifact and names its own parameters in that artifact. A number without a file behind it is not citable — see `docs/TRACEABILITY.md`.
