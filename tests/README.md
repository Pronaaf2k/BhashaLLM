# tests/

```bash
pytest tests/
```

Runs in a few seconds on CPU. No GPU, no model weights, no network — a
test suite that needs a 16 GB card is a test suite nobody runs.

## What is here

| File | What it is | Assertions |
| --- | --- | --- |
| `test_paper_alignment.py` | **The test suite.** Every claim in the paper that can be checked without weights. | 235 |
| `conftest.py` | Tells pytest to skip the diagnostic scripts below. | — |
| `test_download_ocr.py` | Diagnostic: does a HuggingFace OCR dataset download? | 0 |
| `test_eval.py` | Diagnostic: imports `evaluate_ocr_models`, which does not exist. | 0 |
| `test_image_processing.py` | Diagnostic: image preprocessing by eye. | 0 |
| `test_models.py` | Diagnostic: do the three adapters load and generate? | 0 |
| `test_ocr_lang.py` | Diagnostic: does PaddleOCR accept `lang='bn'`? | 0 |
| `test_ocr_libs.py` | Diagnostic: are EasyOCR / PaddleOCR / Tesseract installed? | 0 |
| `test_write.py` | One comment line. | 0 |

The seven diagnostics are named `test_*.py`, which is the pattern pytest
collects, but none of them is a test: no assertions, two take positional
arguments that pytest reads as fixture requests and errors on, and two
import modules that are not importable from the repository root. `pytest`
failed during collection before running anything. `conftest.py` excludes
them; nothing was deleted. Run them directly:

```bash
python tests/test_ocr_libs.py
python tests/test_download_ocr.py
```

Recorded in `docs/ERRATA.md` C25.

## What `test_paper_alignment.py` covers

It is organised by the paper claim each group checks, so a failure names
the claim it breaks rather than the function it broke.

| Group | Checks |
| --- | --- |
| Table III | Every committed phase config resolves to the paper's hyperparameters; the two Phase-3 deviations are present; the pre-Table-III LoRA target set stays reachable |
| Table IV | `epochs_covered` arithmetic; the recovered Phase-1 log's four disagreements with the table; the token count agreeing with the corpus audit |
| Table II | Full fine-tuning reaching ~170 GB; only QLoRA fitting in 16 GB; the trainable counts reproducing 42M and 3.3M; ERRATA B5's multiplier |
| Table VII | The committed OCR predictions scoring far from 12% |
| Sec. III-A | Detection, and the correction stage rejecting rewrites, empty output and Devanagari drift |
| Sec. III-E | Blinding hides identity, order re-randomises per item, the seed reproduces, self-identifying outputs are flagged |
| Sec. III-F | The web UI is opt-in and calls only `/api/v1` |
| Sec. IV-B/IV-C | NFKC and Latin stripping keeping the danda; the by-document split; rare grapheme roots surviving sampling; the dataset shape and path portability |
| Sec. V-B | Correction-rate denominators; an aggressive corrector being penalised |
| Sec. VI-F | The energy arithmetic at both 600 ms and 1450 ms |
| Benchmark | The `llama3.2:latest` tag mismatch; the recovered provenance; the leaked OCR-correction reference |

Three tests skip without `torch` or `fastapi`. That is expected on a
machine that has not installed the full stack.

## Adding a test

Assert against the paper, not against the implementation. A test named
`test_rare_roots_survive_stratified_sampling` that quotes Sec. IV-C in its
docstring tells the next reader *why* the behaviour matters; a test named
`test_allocate_returns_dict` does not.
