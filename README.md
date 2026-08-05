# BhashaLLM

A Bangla text-generation and handwritten-OCR stack that runs end to end on
a single 16 GB consumer GPU. It pairs a QLoRA-fine-tuned instruction model
with a vision-language OCR pipeline, and benchmarks nine language and
vision-language architectures (1.5B–12B parameters) on Bangla translation,
summarisation and OCR correction.

This repository is the artifact accompanying the paper below. If you are
here to check a number in that paper, start with
[`docs/TRACEABILITY.md`](docs/TRACEABILITY.md), which maps each claim to
the file and command that produces it, and
[`docs/ERRATA.md`](docs/ERRATA.md), which records where the manuscript and
this repository do not agree.

> **BhashaLLM: A QLoRA-Based Framework for Bangla Text Generation and
> Handwritten Character Recognition.**
> S. A. Binaaf, M. Y. Arafat, A. S. Chaklader, R. M. Rahman.
> Department of Electrical and Computer Engineering, North South
> University, Dhaka, Bangladesh.
> <!-- TODO: venue, year, DOI once the proceedings are published -->

---

## Read this before citing a number

The manuscript is published and fixed. This repository is not, and several
figures in the paper need qualifiers they did not receive in print. The
substantive ones:

- **No 11-billion-parameter model was fine-tuned.** Llama-3.2-11B was
  evaluated by quantised inference. QLoRA training was applied to the 1.5B
  instruction model and the vision-language OCR model. Table II of the
  paper is a feasibility analysis, not a record of a run.
- **The software versions in Section IV-A are wrong** and describe an
  environment that could not have run this project. The correct versions
  are below and in `docs/environment_capture.txt`.
- **The 12% OCR CER is measured on handwriting the model has seen.** The
  self-collected split is not writer-disjoint. Treat it as an estimate for
  familiar handwriting.
- **Table VI's ROUGE figures predate any committed ROUGE implementation**
  and their tokenizer is unrecorded. Bangla ROUGE is unusually easy to get
  silently wrong; see `eval/text_metrics.py`.

Full list with reasoning: [`docs/ERRATA.md`](docs/ERRATA.md).

---

## Environment

Verified on:

| | |
| --- | --- |
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM (Blackwell, sm_120), 200 W cap |
| CPU / RAM | 12-core Intel Core i9, 32 GB |
| PyTorch | 2.10.0 |
| Transformers | 4.57.6 |
| BitsAndBytes | 0.49.1 |
| PEFT | 0.18.1 |
| CUDA runtime | 12.8.x |

Do not transcribe this table into a paper by hand. Regenerate it:

```bash
bash scripts/capture_environment.sh    # writes docs/environment_capture.txt
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # direct dependencies
# pip install -r requirements-full.lock  # exact frozen environment
```

`requirements.txt` lists what the pipeline imports. `requirements-full.lock`
is the full `pip freeze` from the development machine, preserved for exact
reproduction. It contains packages this project does not use — see
`docs/ERRATA.md` group C.

## Models

Base models and adapters total about 10.5 GB and are not in git. Layout
under `models/`:

```
models/
  base_models/
    Qwen2.5-1.5B-Instruct/     2.9 GB   Qwen/Qwen2.5-1.5B-Instruct
    Bangla-OCR-SFT/            4.0 GB   swapnillo/Bangla-OCR-SFT
  bangla_adapters/final_adapter/          1.5 GB   phase 1
  instruct_adapters/final_instruct_adapter/ 1.4 GB phase 2
  ocr_adapters/<adapter>/                 743 MB   phase 3
```

Only the base model and the currently needed adapter are resident at
runtime, so the 6.9 GB of base weights is not duplicated per task.

> The Phase-3 adapter directory is currently named `banglawriting_adapter`,
> which does not match the training data described in the paper. This is
> unresolved — see `docs/ERRATA.md` §C1. Do not rely on the OCR
> training-data description until it is.

## Running it

The entry point is `main.py`, which serves the FastAPI application:

```bash
python main.py                     # or: uvicorn main:app --reload
```

Section III-F of the paper describes `test_models.py` as the primary
production interface. That script is not in this repository; `main.py` is
the entry point. Recorded in `docs/ERRATA.md` group C.

## Reproducing the paper

Each command writes a JSON file that contains the numbers behind the
corresponding table. Regenerated figures supersede the printed ones where
they differ.

| Paper location | Command | Output |
| --- | --- | --- |
| Sec. III-B — base model selection | `python eval/compute_bpc.py --models Qwen/Qwen2.5-1.5B-Instruct facebook/xglm-1.7b --corpus data/splits/test.txt` | `eval/bpc_comparison.json` |
| Sec. IV-A — environment | `bash scripts/capture_environment.sh` | `docs/environment_capture.txt` |
| Table IV — training phases | `python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml` (likewise phases 2, 3) | `logs/phase{1,2,3}_summary.json` |
| Table VI — summarisation | `python eval/text_metrics.py --pred benchmarks/raw/<model>.jsonl` | `eval/rouge_results.json` |
| Sec. V-A — BLEU / chrF++ | same command; requires `sacrebleu` | signatures included in output |
| Sec. V-C — script integrity | `python eval/script_integrity.py benchmarks/raw/*.jsonl --out eval/script_integrity.json` | `eval/script_integrity.json` |
| Table VII — OCR CER | `python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl --manifest data/handwriting/manifest.csv --group-by writer_id` | `eval/ocr_cer.json` |
| Sec. V-B — grapheme breakdown | same command | `by_grapheme_category` in the same file |
| Table V — human evaluation | `python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv` | `human_eval/alpha_by_dimension.json` |

Training runs log peak VRAM and computed epoch coverage, so Table II and
Table IV are read out of the logs rather than typed:

```python
import torch, json, time
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
# ... training loop ...
json.dump({
    "phase": "1_bangla_pt",
    "seed": cfg["seed"],
    "steps": step,
    "final_train_loss": loss,
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    "wall_clock_s": time.time() - t0,
    "n_train_sequences": len(train_ds),
    "tokens_per_sequence": cfg["max_seq_length"],
    "effective_batch": cfg["batch_size"] * cfg["gradient_accumulation_steps"],
    "epochs_covered": (step * cfg["batch_size"]
                       * cfg["gradient_accumulation_steps"]) / len(train_ds),
}, open("logs/phase1_summary.json", "w"), indent=2)
```

`epochs_covered` is computed deliberately. Section IV-C's 0.8-epoch figure
does not reconcile with the stated corpus size (`docs/ERRATA.md` §B2);
whatever this line prints is the number to use.

## Two Bangla evaluation hazards

Both were found while writing the scripts in `eval/`, and both silently
corrupt results rather than failing loudly.

**The danda is not Devanagari-only.** U+0964 and U+0965 sit in the
Devanagari block but are shared Indic punctuation and the correct sentence
terminators in Bangla. A script-confusion detector that flags any
codepoint in U+0900–U+097F marks every correctly punctuated Bangla
sentence as confused. `eval/script_integrity.py` exempts them.

**`\w` drops Bangla vowel signs.** Python's `\w` matches only what
`str.isalnum()` accepts, and `isalnum()` is False for Unicode categories Mn
and Mc — which is what Bangla matra and the hasant are. A `[^\W_]+`
tokenizer splits `বাংলাদেশের` into `ব`, `ল`, `দ`, `শ`, `র`. Separately, the
google-research `rouge_score` default tokenizer strips all non-ASCII and
returns 0.0 on any Bangla input. `eval/text_metrics.py` uses a
mark-aware tokenizer and prints its name in every output file.

## Data

See [`DATA_CARD.md`](DATA_CARD.md) for provenance, licensing and
collection procedure.

The self-collected handwriting set is 1,500 pages from **three** writers,
which is enough to show that fine-tuning helps and not enough to establish
how far the result generalises. Per-writer identifiers are required for
writer-disjoint evaluation; see the data card for status.

Text corpora are drawn from public-domain literary archives and Kaggle
datasets, each carrying its own license. The data card lists URL, access
date, license and checksum per source.

## Layout

```
bhasha/            core package (app, data, eval, llm, ocr, scripts, utils)
configs/           one YAML per training phase
eval/              metric implementations; each writes a JSON artifact
benchmarks/        per-model raw generations and the model registry
human_eval/        anchored rubric, anonymised ratings, reliability
logs/              per-phase training summaries
docs/              errata, traceability register, environment capture
data/              splits, checksums, handwriting manifest
tests/             mirrors the bhasha/ hierarchy
main.py            FastAPI entry point
```

## Limitations

Carried from the paper and not resolved here: script confusion persists in
the smaller models (up to 23% of Qwen-1.5B generations); cultural and
historical context in Bangla literature is handled poorly by models trained
primarily on English; factual hallucination is unresolved across every
model tested; the 6.6M-token literary corpus is not balanced across genre,
register or period; OCR degrades on stylised handwriting, poor image
quality and overlapping strokes; the OCR test set is not writer-disjoint;
all training runs used a single seed.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)).
Self-collected handwriting dataset: CC BY 4.0 — see `DATA_CARD.md`.
Third-party datasets and base models retain their own licenses.
