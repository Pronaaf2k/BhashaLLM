"""pytest configuration for BhashaLLM.

Running `pytest` on this directory used to fail before it ran anything.
Seven files sit here; only one is a test suite. The rest are manual
diagnostic scripts that were written as `test_*.py`, which is exactly the
name pattern pytest collects, so pytest tries to run them and errors:

* `test_ocr_lang.py` and `test_ocr_libs.py` define `test_lang(lang_code)`
  and `test_ocr_libraries(image_path)`. pytest reads a positional argument
  on a test function as a fixture request, finds no fixture by that name,
  and reports `fixture 'lang_code' not found` — a collection error, not a
  failure.
* `test_eval.py` does `from evaluate_ocr_models import compute_metrics`,
  and no such module exists anywhere in the repository.
* `test_models.py` and `test_image_processing.py` do
  `from model_paths import ...` rather than
  `from bhasha.scripts.model_paths import ...`, so they import only when
  the working directory happens to be `bhasha/scripts/`.
* `test_write.py` contains one line: `# test`.
* None of the six contains a single `assert`.

The scripts are useful — they check whether PaddleOCR speaks Bengali,
whether EasyOCR and Tesseract are installed, whether a dataset downloads.
They are just not tests, and none of them is removed here. This file tells
pytest to leave them alone so that `pytest` runs the one real suite
cleanly. Run the diagnostics directly instead:

    python tests/test_ocr_libs.py
    python tests/test_download_ocr.py

See `tests/README.md` and `docs/ERRATA.md` C25.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Legacy diagnostic scripts, ignored by collection. Removing an entry from
# this list is the right move once that file has been converted into a real
# test with assertions.
LEGACY_DIAGNOSTIC_SCRIPTS = [
    "test_download_ocr.py",     # downloads a HF dataset; no assertions
    "test_eval.py",             # imports the absent `evaluate_ocr_models`
    "test_image_processing.py", # unqualified `model_paths` import
    "test_models.py",           # unqualified `model_paths` import
    "test_ocr_lang.py",         # test_lang(lang_code) -> fixture error
    "test_ocr_libs.py",         # test_ocr_libraries(image_path) -> same
    "test_write.py",            # one comment line
]

collect_ignore = list(LEGACY_DIAGNOSTIC_SCRIPTS)
