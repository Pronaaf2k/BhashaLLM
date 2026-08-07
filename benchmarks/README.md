# benchmarks/

Raw generations, the evaluation item sets, and the record of how each model
was served. `docs/TRACEABILITY.md` points four rows at this directory.

```
benchmarks/
  model_registry.json      how each of the nine models was quantised and served
  items/
    translation.jsonl      fixed item set for Table V
    summarisation.jsonl    fixed item set for Table VI
    ocr_correction.jsonl   fixed item set for Sec. V-B
  raw/
    <model>.jsonl          one file per model, one record per generation
  latency.json             written by eval/latency.py
```

`raw/` and the populated item sets are not committed yet — see the status
column in `docs/TRACEABILITY.md`. The schemas below are what the evaluation
scripts in `eval/` read, so anything written to these paths in this format
flows straight through to the JSON artifacts the paper's tables are read
from.

## Why the item set has to be fixed and committed

Tables V and VI compare nine models. If each model was scored on a
different set of prompts, or on a set that was regenerated between runs,
the comparison is not a comparison. `docs/TRACEABILITY.md` lists
"Evaluation item set and N" as `MISSING`, and neither table states N.
Committing `items/*.jsonl` fixes both: N becomes readable, and a reader can
re-score any model on exactly the inputs the paper used.

## Schemas

### `items/translation.jsonl` and `items/summarisation.jsonl`

```json
{"item_id": "trans_001", "source": "<English source text>", "reference": "<Bangla reference>", "task": "translation"}
```

`source` is the input given to the model. `reference` is the human
reference used by `eval/text_metrics.py` for ROUGE, BLEU and chrF++.

### `items/ocr_correction.jsonl`

```json
{"item_id": "corr_001", "noisy": "<raw OCR output>", "reference": "<ground truth>"}
```

Scored by `eval/ocr_correction.py` once a model's `hypothesis` is added.

### `raw/<model>.jsonl`

One record per generation. The union of the keys every evaluation script
reads, so a single file per model feeds all of them:

```json
{
  "item_id": "trans_001",
  "model": "llama-3.2-11b",
  "task": "translation",
  "prompt": "<exact prompt sent, Bangla-only per Sec. IV-C>",
  "output": "<raw generation>",
  "hypothesis": "<same as output; the key eval/text_metrics.py reads>",
  "reference": "<from the item set>",
  "token_logprobs": [-0.01, -0.42],
  "n_new_tokens": 100,
  "latency_ms": 1450.0,
  "max_new_tokens": 100
}
```

| Key | Read by |
| --- | --- |
| `output` | `eval/script_integrity.py` |
| `hypothesis`, `reference` | `eval/text_metrics.py`, `eval/ocr_correction.py` |
| `token_logprobs` | `eval/confidence.py` |
| `latency_ms`, `max_new_tokens` | cross-check against `latency.json` |

`max_new_tokens` is mandatory on any record carrying `latency_ms`. A
latency without its token budget is not interpretable —
`docs/ERRATA.md` B3.

## Regenerating the artifacts

```bash
python eval/script_integrity.py benchmarks/raw/*.jsonl --out eval/script_integrity.json
python eval/text_metrics.py     --pred benchmarks/raw/llama-3.2-11b.jsonl --out eval/rouge_results.json
python eval/ocr_correction.py   --pred benchmarks/raw/ocr_correction.jsonl --out eval/ocr_correction.json
python eval/confidence.py       --pred benchmarks/raw/llama-3.2-11b.jsonl --out eval/confidence.json
python eval/latency.py --mode local --models Qwen/Qwen2.5-1.5B-Instruct \
    --max-new-tokens 100 --load-4bit --out benchmarks/latency.json
python eval/energy.py --from-latency-json benchmarks/latency.json \
    --grid-factor <cite it> --grid-source "<source, access date>" --out eval/energy.json
```

## Before citing any cross-model number

Fill every `FILL` field in `model_registry.json`. Five of the nine models
cannot be served in fp16 on 16 GB and were therefore quantised, and the
manuscript records the format for none of them. Quantisation moves both
output quality and script integrity, so it is a confound that runs through
Tables V, VI and VII rather than a footnote — `docs/ERRATA.md` B10.
