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

- **The benchmarked "Llama-3.2-11B" was almost certainly a 3B model.**
  `bhasha/llm/run_benchmark_suite.py` maps that row to the Ollama tag
  `llama3.2:latest`, which resolves to `llama3.2:3b` — 3.21B parameters,
  Q4_K_M, a 2.0 GB download. The 11B model is Llama-3.2-11B-**Vision** and
  is served under a different tag. This affects the paper's central
  conclusion and every figure attributed to that model.
  [`docs/ERRATA.md`](docs/ERRATA.md) §A00.

- **The nine-model benchmark ran on Ollama, not the paper's stack.** GGUF
  at Q4_K_M via llama.cpp, not 4-bit NF4 via BitsAndBytes; temperature 0.3,
  not the 0.7 of Sec. IV-C; and **one prompt per task**, so Tables V and VI
  rest on N=1 per model. §C11–C13.

- **The OCR-correction prompt contains its own answer.** Every model was
  shown `(Expected: ...)` inside the prompt, so Sec. V-B's correction rates
  measure copying. §C17.

- **No 11-billion-parameter model was fine-tuned.** QLoRA training was
  applied to the 1.5B instruction model and the vision-language OCR model.
  Table II of the paper is a feasibility analysis, not a record of a run.
- **The software versions in Section IV-A are wrong** and describe an
  environment that could not have run this project. The correct versions
  are below and in `docs/environment_capture.txt`.
- **The 12% OCR CER is measured on handwriting the model has seen.** The
  self-collected split is not writer-disjoint. Treat it as an estimate for
  familiar handwriting.
- **Table VI's ROUGE figures predate any committed ROUGE implementation**
  and their tokenizer is unrecorded. Bangla ROUGE is unusually easy to get
  silently wrong; see `eval/text_metrics.py`.

- **The 6.6M-token corpus is not in this repository.** The committed
  `text dataset.rar` is 58 KB compressed and holds ~110 numbered text
  files with no author metadata, so neither Table IV's corpus size nor
  Section IV-C's "no work by the same author appears on both sides" can be
  reproduced from it. Measure it yourself with
  `python scripts/audit_text_corpus.py`; findings in
  [`docs/ERRATA.md`](docs/ERRATA.md) §C8.

- **The OCR adapter was trained on BanglaWriting, not Ekush.** Section
  IV-C describes Phase 3 as Ekush plus self-collected pages. The Phase-3
  loader, the OCR evaluation script and the production model-path resolver
  all point at `data/processed/banglawriting`. BanglaWriting is not cited
  anywhere in the paper. This was the repository's highest-priority open
  item and is now resolved: [`docs/ERRATA.md`](docs/ERRATA.md) §A0.

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

Two interfaces. Section III-F of the paper names `test_models.py` as the
primary production interface; `main.py` serves the HTTP API.

### `test_models.py` — the CLI (paper Sec. III-F)

Applies ChatML formatting for the grading model and the correct resizing
and normalisation for the OCR model automatically, so you do not need to
know either model's input format. Only the base model and the currently
needed adapter are held in memory.

```bash
python test_models.py status                       # what is loaded, VRAM, adapters on disk

python test_models.py generate --adapter bangla \
    --prompt "বাংলা সাহিত্যের ইতিহাস সম্পর্কে লিখুন।"

python test_models.py grade \
    --question "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?" \
    --reference "তিনি একজন বাঙালি কবি ও সাহিত্যিক।" \
    --answer   "তিনি একজন লেখক।"

python test_models.py ocr --image page.png --confidence
python test_models.py chat                         # REPL; /adapter <name> swaps live
```

Batch OCR writes the JSONL the evaluation scripts read, so inference feeds
straight into `eval/`:

```bash
python test_models.py ocr --manifest data/handwriting/manifest.csv \
    --split test --confidence --out eval/ocr_predictions.jsonl
```

### The hybrid OCR pipeline (paper Sec. III-A)

Detection → QLoRA Qwen-VL recognition → LLM correction, the three stages of
the paper's third contribution:

```bash
python -m bhasha.ocr.hybrid_pipeline --image page.png
python -m bhasha.ocr.hybrid_pipeline --image page.png --detector paddle
python -m bhasha.ocr.hybrid_pipeline --image line.png --detector none --no-correct
```

Batch mode writes `noisy` (pre-correction) and `hypothesis`
(post-correction) into one file, so the recognizer and the corrector are
scored separately — which is what Table VII and Section V-B report
separately:

```bash
python -m bhasha.ocr.hybrid_pipeline --manifest data/handwriting/manifest.csv \
    --split test --out eval/ocr_predictions.jsonl
python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl \
    --manifest data/handwriting/manifest.csv --group-by writer_id
python eval/ocr_correction.py --pred eval/ocr_predictions.jsonl
```

For Table VII's **"Before"** column, run the un-adapted base model:

```bash
python -m bhasha.ocr.hybrid_pipeline --manifest ... --no-adapter --no-correct
python test_models.py ocr --manifest ... --no-adapter --confidence
```

`bhasha/ocr/pipeline.py` is the repository's original PaddleOCR+Tesseract
pipeline. It is unchanged and still works, but its correction stage is a
placeholder and its recognizer is not the paper's — see
`docs/ERRATA.md` §C4.

### `main.py` — the API

```bash
python main.py                     # or: uvicorn main:app --reload
```

Serves two surfaces:

| Prefix | What it is |
| --- | --- |
| `/api/v1/*` | The pipeline described in the paper. `status`, `generate`, `grade`, `ocr`, `adapter`. Fully local, one resident adapter, Sec. IV-C decoding. |
| `/api/analyze`, `/api/chat`, `/api/philosophical` | The original application: a ResNet-34 three-head grapheme classifier plus Gemini cloud calls. Retained and unchanged. |

The optional web frontend of Sec. III-F is **opt-in**, because the paper
says it "is not loaded unless explicitly opened":

```bash
BHASHA_ENABLE_UI=1 python main.py     # then open http://localhost:5000/ui
```

It is one static file that calls the same `/api/v1` endpoints as the CLI —
no build step, no npm, no framework.

The legacy endpoints are not part of the paper's methodology and two of
them make outbound network calls, which sits awkwardly beside the offline
deployment claim in Sections III-F and VI-F. Recorded in
`docs/ERRATA.md` §C3. Reproduce Section III-F against `/api/v1` or the
CLI.

## Reproducing the paper

Each command writes a JSON file that contains the numbers behind the
corresponding table. Regenerated figures supersede the printed ones where
they differ.

| Paper location | Command | Output |
| --- | --- | --- |
| Sec. III-A — hybrid OCR pipeline | `python -m bhasha.ocr.hybrid_pipeline --image page.png` | detection → recognition → correction |
| Sec. III-C / Table II — memory budget | `python eval/memory_budget.py --table-ii --out eval/memory_budget.json` | `eval/memory_budget.json` |
| Table II — the OOM row | `python scripts/capture_oom_attempt.py` | `logs/oom_attempt.txt` |
| Sec. III-E — blind rating sheets | `python eval/make_rating_sheets.py --pred benchmarks/raw/*.jsonl --seed 42` | `human_eval/{rating_sheet.csv,rating_sheet.md,blinding_map.json}` |
| Sec. III-F — local footprint | `bash scripts/capture_footprint.sh` | `docs/footprint.txt` |
| Sec. IV-B — tokeniser sanity checks | `python eval/tokenizer_sanity.py --models Qwen/Qwen2.5-1.5B-Instruct facebook/xglm-1.7b` | `eval/tokenizer_sanity.json` |
| Tables V, VI — recover the committed generations | `python scripts/convert_llm_outputs.py` | `benchmarks/raw/*.jsonl` from `llm outputs/*.md` |
| Sec. IV-B — audit the corpus archive | `python scripts/audit_text_corpus.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct` | `data/text_corpus_audit.json`, `data/text_raw.sha256` |
| Sec. IV-B — corpus preprocessing and splits | `python -m bhasha.data.text_corpus --input data/raw/nazrul data/raw/tagore --out-dir data/splits --tokenizer Qwen/Qwen2.5-1.5B-Instruct` | `data/splits/{train,val,test}.txt`, `split_manifest.json` |
| Sec. IV-C — Ekush stratified sample | `python -m bhasha.data.ekush_sampling --input data/processed/ekush_prepared/train.jsonl --n 6000 --out data/processed/ekush_sampled_6000.jsonl` | sample + stratification report |
| Sec. IV-B / VI-D — handwriting manifest | `python -m bhasha.data.manifest --validate data/handwriting/manifest.csv` | writer-disjointness audit |
| Sec. III-B — base model selection | `python eval/compute_bpc.py --models Qwen/Qwen2.5-1.5B-Instruct facebook/xglm-1.7b --corpus data/splits/test.txt` | `eval/bpc_comparison.json` |
| Sec. IV-A — environment | `bash scripts/capture_environment.sh` | `docs/environment_capture.txt` |
| Table IV — training phases | `python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml` (then `bhasha.llm.train_instruct` and `bhasha.ocr.train` with phases 2 and 3) | `logs/phase{1,2,3}_summary.json` |
| Table VI — summarisation | `python eval/text_metrics.py --pred benchmarks/raw/<model>.jsonl` | `eval/rouge_results.json` |
| Sec. V-A — BLEU / chrF++ | same command; requires `sacrebleu` | signatures included in output |
| Sec. V-C — script integrity | `python eval/script_integrity.py benchmarks/raw/*.jsonl --out eval/script_integrity.json` | `eval/script_integrity.json` |
| Table VII — OCR CER | `python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl --manifest data/handwriting/manifest.csv --group-by writer_id` | `eval/ocr_cer.json` |
| Sec. V-B — grapheme breakdown | same command | `by_grapheme_category` in the same file |
| Sec. V-B — OCR correction rates | `python eval/ocr_correction.py --pred benchmarks/raw/ocr_correction.jsonl` | `eval/ocr_correction.json`, every denominator reported |
| Table VII — confidence score | `python eval/confidence.py --pred eval/ocr_predictions.jsonl` | `eval/confidence.json` + calibration error |
| Sec. V-C — latency | `python eval/latency.py --mode local --models <ids> --max-new-tokens 100 --load-4bit` | `benchmarks/latency.json` |
| Sec. VI-F — energy / CO2 | `python eval/energy.py --from-latency-json benchmarks/latency.json --grid-factor <x> --grid-source "<cite>"` | `eval/energy.json` |
| Table V — human evaluation | `python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv` | `human_eval/alpha_by_dimension.json` |
| Sec. IV–V — how each model was served | fill the `FILL` fields | `benchmarks/model_registry.json` |

Two of these refuse to run without an argument the paper omitted, on
purpose. `eval/latency.py` requires `--max-new-tokens`, because Section
V-C's figures imply a 100-token budget while Section IV-C fixes decoding at
256 and the paper never says which applies (`docs/ERRATA.md` B3).
`eval/energy.py` requires `--grid-factor` and `--grid-source`, because the
20 g CO2 figure implies an uncited carbon intensity below published
Bangladesh values (`docs/ERRATA.md` B4). Both would rather fail than emit
a number with an anonymous constant behind it.

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

## Training

All three phases read their configuration from `configs/`, which carries
the paper's Table III verbatim plus the two Phase-3 deviations Section IV-C
documents (2,000 steps rather than 500; learning rate 1e-4 rather than
2e-4).

```bash
python -m bhasha.llm.train          --config configs/phase1_bangla_pt.yaml
python -m bhasha.llm.train_instruct --config configs/phase2_grading_sft.yaml
python -m bhasha.ocr.train          --config configs/phase3_ocr_sft.yaml
```

Print a resolved configuration without training anything:

```bash
python -m bhasha.config --config configs/phase1_bangla_pt.yaml
python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml --print-config
```

Resolution order is **command-line flag > `--config` YAML > Table III
default**, so every hyperparameter has one visible source. Each run writes
`logs/phase{1,2,3}_summary.json` containing the measured peak VRAM, the
final losses, the exact hyperparameters, the captured library versions, and
a **computed** `epochs_covered`. That last field exists because Section
IV-C's 0.8-epoch figure does not reconcile with the stated corpus size
(`docs/ERRATA.md` B2); whatever the log prints is the number to cite.

**Defaults changed to match Table III.** The trainers previously targeted
all seven projection matrices, used `paged_adamw_32bit`, set no learning
rate schedule and no seed, and used a 1024-token context. Table III
specifies `q_proj, v_proj` — which is what yields Table II's 9.4M trainable
parameters — with AdamW, cosine decay after 50 warm-up steps, seed 42 and
512 tokens. To reproduce the previous behaviour exactly:

```bash
python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml \
    --target-modules legacy --optim paged_adamw_32bit
```

Every original command-line flag still works.

## Layout

```
bhasha/            core package (app, data, eval, llm, ocr, scripts, utils)
  config.py        Table III loader; makes configs/*.yaml authoritative
  data/            OCRDataset, corpus preprocessing, Ekush sampling, manifest
  app/             FastAPI app, adapter manager, /api/v1 router, opt-in /ui
  ocr/             legacy Paddle pipeline + the paper's hybrid pipeline
  utils/           run summaries (peak VRAM, epochs_covered), helpers
scripts/           environment, footprint, OOM and corpus-archive capture
configs/           one YAML per training phase
eval/              metric implementations; each writes a JSON artifact
benchmarks/        per-model raw generations, item sets, and the model registry
human_eval/        anchored rubric, anonymised ratings, reliability
logs/              per-phase training summaries
docs/              errata, traceability register, environment capture
data/              splits, checksums, handwriting manifest
tests/             mirrors the bhasha/ hierarchy
test_models.py     CLI — the primary production interface (paper Sec. III-F)
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
