# Traceability register

One row per numeric claim in the paper. Each row names the evidence file
that should contain that number and the command that produces it.

The standard: a reader should be able to open this repository, find one
file, and see the number — together with the script that produced it, the
seed, the input data, and the date it ran.

**Status codes**

| Code | Meaning |
| --- | --- |
| `TRACED` | Evidence file committed; number is readable from it. |
| `CHECK` | Evidence may already exist in `logs/`, `report/` or `llm outputs/`. Verify before re-running — this is the cheapest work in the whole plan. |
| `MISSING` | No evidence file exists yet. Regenerate with the listed command. |
| `CORRECTED` | The published figure is wrong or unqualified; `docs/ERRATA.md` carries the correction. |

Update the status column as files land. The register is worth nothing if
it is not maintained.

---

## Environment and memory

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Software stack versions | Sec. IV-A | `docs/environment_capture.txt` | `bash scripts/capture_environment.sh` | `CORRECTED` (ERRATA A1) |
| Hardware: RTX 5070 Ti, 16 GB | Sec. IV-A | same | same | `MISSING` |
| QLoRA peak 9.4 GB | Table II | `logs/phase*_summary.json` → `peak_vram_gb` | any phase run; written by `bhasha/utils/run_summary.py` | `MISSING` (writer now implemented) |
| LoRA r=16 fp16 OOM | Table II | `logs/oom_attempt.txt` (the traceback itself) | `python scripts/capture_oom_attempt.py` | `MISSING` (capture script now implemented) |
| 11B model *trained* on 16 GB | Abstract, Tbl II, III-C, VI-E | `logs/llama11b_train_summary.json` — or claim withdrawn | — | `CORRECTED` (ERRATA B1) |
| Full FT ≈ 170 GB | Sec. III-C | `eval/memory_budget.json` | `python eval/memory_budget.py --table-ii --out eval/memory_budget.json` | `CORRECTED` — computed 171.6 GB; the paper's total is right, its 4x multiplier is not (ERRATA B5) |
| Local footprint 10.5 GB, 66/34/1 | Sec. III-F | `docs/footprint.txt` | `bash scripts/capture_footprint.sh` | `MISSING` (capture script now implemented) |

## Data

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Corpus 6.6M tokens, 80/10/10 by document | Sec. IV-B | `data/splits/{train,val,test}.txt` + `split_manifest.json` | `python -m bhasha.data.text_corpus --input <raw> --out-dir data/splits --tokenizer Qwen/Qwen2.5-1.5B-Instruct` | `MISSING` (split script now implemented) |
| 1,500 self-collected pages, 3 writers | Sec. IV-B | `DATA_CARD.md` + `data/handwriting/manifest.csv` | manual | `MISSING` |
| Per-writer identifiers | Sec. VII | `data/handwriting/manifest.csv` → `writer_id` | `python -m bhasha.data.manifest --template ...` then fill; `--validate` audits writer disjointness | `MISSING` (schema + validator now implemented) |
| Text corpus provenance and licences | Refs [21]–[27] | `DATA_CARD.md` provenance table | manual | `MISSING` |
| Corpus archive contents and size | Sec. IV-B, Table IV | `data/text_corpus_audit.json`, `data/text_raw.sha256` | `python scripts/audit_text_corpus.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct` | `CORRECTED` — the committed archive is 58 KB against a claimed 6.6M tokens (ERRATA C8) |
| Author-disjoint 80/10/10 split | Sec. IV-C | `data/splits/split_manifest.json` → `grouping` | `python -m bhasha.data.text_corpus --group-by author` | `CORRECTED` — the archive carries no author metadata, so the guarantee is unverifiable (ERRATA C8) |
| OCR trained on Ekush + self-collected | Sec. IV-C | `bhasha/ocr/train.py`, `bhasha/eval/ocr_models.py`, `bhasha/scripts/model_paths.py` | loader read | `CORRECTED` — **trained on BanglaWriting; ERRATA A0** |

## Training

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Phase 1: 500 steps, loss 1.31 | Table IV | `logs/phase1_summary.json` | `python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml` | `CHECK` (command now works) |
| Phase 1: 0.8 epochs | Table IV, Sec. IV-C | same → `epochs_covered` | computed, not typed | `CORRECTED` (ERRATA B2) |
| Phase 2: 500 steps, val loss 0.018 | Table IV | `logs/phase2_summary.json` | `python -m bhasha.llm.train_instruct --config configs/phase2_grading_sft.yaml` | `CHECK` (command now works) |
| Phase 3: 2000 steps, val loss 0.31 | Table IV | `logs/phase3_summary.json` | `python -m bhasha.ocr.train --config configs/phase3_ocr_sft.yaml` | `CHECK` (command now works) |
| Single seed, no variance | Sec. IV-C | `seed` field in each summary | — | `TRACED` (disclosed in paper) |

## Model selection and benchmark

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Qwen PPL 3.8 vs XGLM 73.15 | Sec. III-B | `eval/bpc_comparison.json` | `python eval/compute_bpc.py --models ... --corpus data/splits/test.txt` | `CORRECTED` (ERRATA B6) |
| Quantisation of the 9 models | Sec. IV–V | `benchmarks/model_registry.json` | fill the `FILL` fields per model | `CHECK` — file added with schema; fields unfilled (ERRATA B10) |
| Raw generations, 9 models | Tables V, VI | `benchmarks/raw/{model}.jsonl` | benchmark run | `CHECK` |
| Evaluation item set and N | Tables V, VI | `benchmarks/items/{translation,summarisation}.jsonl` | fix N and commit; schema in `benchmarks/README.md` | `MISSING` (schema now specified) |

## Results — text

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Translation scores, 14/15 top | Table V | `human_eval/ratings.csv` | `python eval/make_rating_sheets.py --pred benchmarks/raw/*.jsonl` → rate → `--unblind` → `eval/aggregate_human_eval.py` | `MISSING` (sheet generator now implemented) |
| Inter-rater agreement | Sec. III-E, VI-D | `human_eval/alpha_by_dimension.json` | same | `MISSING` |
| Rubric anchors | Sec. III-E | `human_eval/rubric.md` | — | `TRACED` |
| Blinding map and seed | Sec. III-E | `human_eval/blinding_map.json` | `python eval/make_rating_sheets.py --pred ... --seed 42` | `MISSING` (generator now implemented) |
| ROUGE-1/2/L, 9 models | Table VI | `eval/rouge_results.json` | `python eval/text_metrics.py --pred benchmarks/raw/<model>.jsonl` | `CORRECTED` (ERRATA B8) |
| ROUGE tokenizer for Bangla | Table VI | `tokenizer` field in the same file | same | `TRACED` |
| BLEU / chrF++ | Sec. V-A | `eval/rouge_results.json` → `bleu`/`chrf++` + signatures | `python eval/text_metrics.py --pred ...` (sacrebleu now in requirements.txt) | `MISSING` |
| Script confusion 23% / <1% / >99% | Abstract, Sec. V-C | `eval/script_integrity.json` | `python eval/script_integrity.py benchmarks/raw/*.jsonl` | `MISSING` (definition now `TRACED`, ERRATA B9) |
| Latency 1450 / 680 ms, 69 / 147 tok/s | Sec. V-C | `benchmarks/latency.json` at a stated token budget | `python eval/latency.py --max-new-tokens 100 ...` (budget is a required flag) | `CORRECTED` (ERRATA B3) |
| 42 Wh / 20 g CO2 per 1000 inferences | Sec. VI-F | `eval/energy.json` | `python eval/energy.py --from-latency-json benchmarks/latency.json --grid-factor <x> --grid-source "<cite>"` | `CORRECTED` (ERRATA B4) |

## Results — OCR

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| CER 28% → 12% | Abstract, Table VII | `eval/ocr_cer.json` | `python test_models.py ocr --manifest ... --split test --out eval/ocr_predictions.jsonl` then `python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl` | `MISSING` (inference CLI now implemented) |
| Writer-disjoint CER | Sec. VI-D | same, `by_writer_id` | add `--manifest ... --group-by writer_id` | `MISSING` |
| Per-grapheme accuracy 94/91/85/79% | Sec. V-B | same, `by_grapheme_category` | same command | `CORRECTED` (ERRATA B7) |
| Confidence 0.68 → 0.82 | Table VII | `eval/confidence.json` | `python eval/confidence.py --pred <preds with token_logprobs>` | `CORRECTED` — definition `exp(mean log p)` now in code (ERRATA B11) |
| OCR correction 88% / 85% / 35% | Sec. V-B | `eval/ocr_correction.json` with denominators | `python eval/ocr_correction.py --pred benchmarks/raw/ocr_correction.jsonl` | `CORRECTED` — denominators now defined in code (ERRATA B11) |
| Maung et al. CER 10.37% | Table VII | their 2.47% final figure must appear too | — | `CORRECTED` (ERRATA A3) |

## Architecture and pipeline

| Claim | Paper | Evidence file | Command | Status |
| --- | --- | --- | --- | --- |
| Optional web frontend | Sec. III-F | `bhasha/app/static/index.html` | `BHASHA_ENABLE_UI=1 python main.py` → `/ui` | `TRACED` (was absent, ERRATA C6) |
| Detection decoupled from recognition | Sec. III-A | `bhasha/ocr/hybrid_pipeline.py` | `--detector projection\|paddle\|none` | `TRACED` (was unimplemented, ERRATA C4) |
| LLM correction stage exists | Sec. I contrib. 3, V-B | `bhasha/ocr/hybrid_pipeline.py` → `correct_line` | `python -m bhasha.ocr.hybrid_pipeline --image ...` | `TRACED` (was a stub, ERRATA C4) |
| Table VII "Before" (28% CER, 0.68) | Table VII | `eval/ocr_cer.json` from an un-adapted run | `python test_models.py ocr --no-adapter ...` | `MISSING` (baseline path now implemented) |
| Retrieval not used | Sec. III-D | — | — | `CORRECTED` — a RAG stack is present but feeds no result (ERRATA C5) |
| Tokeniser sanity checks vs refs [24]–[27] | Sec. IV-B | `eval/tokenizer_sanity.json` | `python eval/tokenizer_sanity.py --models ... --wordlist ... --corpus ...` | `MISSING` (checks now implemented) |
| Subword inefficiency makes context scarce | Sec. III-C | same → `fertility` | same command | `MISSING` (now measurable) |
| Bangla/Devanagari token adjacency | Sec. VI-A | same → `script_adjacency` | same command | `MISSING` — partial only; the full ablation Sec. VI-A describes is still not run |

## References

| Claim | Paper | Status |
| --- | --- | --- |
| ViT arXiv:2010.01192 | Ref [7] | `CORRECTED` → arXiv:2010.11929 (ERRATA A2) |
| TrOCR 2109.10282, Qwen-VL 2308.12966, LoRA 2106.09685, QLoRA 2305.14314, IndicTrans2 2305.16307, activation engineering 2308.10248 | Refs [8][11][13][14][16][31] | `TRACED` — verified correct |
| XGLM cited via model card | Ref [1] | Should cite the XGLM paper for a claim about pre-training coverage |
| Literary corpora, no URL or date | Refs [21], [22] | Not reproducible data citations; see `DATA_CARD.md` |

---

## Implementation status of the producing commands

A `MISSING` status now means only that the evidence file has not been
generated yet. Every command in the tables above exists and runs; before
this release several of them named scripts that were not in the
repository. What is missing is data and compute, not code.

| Producing script | State |
| --- | --- |
| `bhasha/data/text_corpus.py` | added — NFKC, Latin stripping, 80/10/10 by document |
| `bhasha/data/ekush_sampling.py` | added — stratified by grapheme root |
| `bhasha/data/manifest.py` | added — schema, validator, writer-disjointness audit |
| `bhasha/data/dataset.py` | added — restores the import Phase 3 needs |
| `bhasha/config.py` | added — makes `configs/phase*.yaml` authoritative |
| `bhasha/utils/run_summary.py` | added — writes `logs/phase*_summary.json` |
| `eval/ocr_correction.py` | added — correction rates with denominators |
| `eval/confidence.py` | added — defines the Table VII confidence score |
| `eval/latency.py` | added — requires an explicit token budget |
| `eval/energy.py` | added — requires a cited grid factor |
| `benchmarks/model_registry.json` | added — schema present, `FILL` fields open |
| `test_models.py` | added — the Sec. III-F interface, at the repository root |
| `bhasha/app/adapter_manager.py`, `routes_v1.py` | added — Sec. III-F adapter residency, mounted at `/api/v1` |
| `bhasha/ocr/hybrid_pipeline.py` | added — detection → QLoRA recognition → real LLM correction (ERRATA C4) |
| `eval/make_rating_sheets.py` | added — blind rating sheets, blinding map, unblinding |
| `eval/memory_budget.py` | added — Table II term by term |
| `eval/tokenizer_sanity.py` | added — Sec. IV-B sanity checks, fertility, conjunct/matra integrity |
| `scripts/capture_oom_attempt.py` | added — captures `logs/oom_attempt.txt` |
| `scripts/capture_footprint.sh` | added — writes `docs/footprint.txt` |
| `scripts/audit_text_corpus.py` | added — extracts and audits `text dataset.rar` against Table IV |
| `bhasha/app/routes_ui.py`, `static/index.html` | added — the opt-in Sec. III-F frontend |
| `eval/{compute_bpc,ocr_cer,script_integrity,text_metrics,aggregate_human_eval}.py` | already present |

## Before tagging `v1.0-paper`

- [ ] No row still reads `CHECK` — each has been resolved to `TRACED` or `MISSING`
- [x] ERRATA C1 resolved: the OCR training dataset is confirmed by reading the loader — **BanglaWriting, not Ekush; see ERRATA A0**
- [ ] BanglaWriting cited in the reference list and completed in `DATA_CARD.md`
- [ ] `benchmarks/model_registry.json` `FILL` fields completed (ERRATA B10)
- [ ] `benchmarks/items/*.jsonl` committed so N is readable for Tables V and VI
- [ ] `docs/environment_capture.txt` committed
- [ ] `DATA_CARD.md` committed with per-writer identifiers
- [ ] Corpus archive resolved: either the full 6.6M-token corpus is committed, or Table IV's figure is revised (ERRATA C8)
- [ ] Author metadata added to the text corpus, or the Sec. IV-C author-disjointness claim withdrawn (ERRATA C8)
- [ ] `LICENSE` and `CITATION.cff` committed
- [ ] Repository description and topics set on GitHub
- [ ] Tag pushed and archived for a DOI; DOI recorded in `README.md` and `CITATION.cff`
- [ ] Adapter weights published (3.6 GB — too large for git, free to host)
