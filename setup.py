"""Package definition for BhashaLLM.

Kept in step with `requirements.txt`. Where the two disagree,
`requirements.txt` is authoritative — it is the file `README.md` tells
people to install from, and it carries pins.

Three defects were corrected here (see `docs/ERRATA.md` C26):

1. **`PyYAML` was missing.** `bhasha/config.py` imports `yaml` to read
   `configs/phase*.yaml`, which is the entire Table III mechanism. A
   `pip install .` produced an installation where
   `python -m bhasha.llm.train --config ...` failed at import.
2. **`trl` was missing**, so Phase 2 (`bhasha/llm/train_instruct.py`,
   which uses `SFTTrainer`) could not run from an installed copy either.
   Eleven other direct dependencies were absent for the same reason.
3. **No `package_data`.** `bhasha/app/static/index.html` — the Section
   III-F web frontend — and `configs/*.yaml` are not Python modules, so
   `find_packages()` does not carry them and neither shipped in a wheel.
   An installed copy served a 404 at `/ui` and could not load any phase
   config.

The optional groups at the bottom are packages the previous
`install_requires` treated as mandatory. `docs/ERRATA.md` group C records
that PaddleOCR is not part of the methodology in the paper, and that the
Gemini client belongs to the legacy endpoints rather than the pipeline;
forcing either on every install is what made `requirements.txt` and
`setup.py` disagree in both directions.
"""

from setuptools import setup, find_packages

# Direct dependencies, mirroring requirements.txt. Unpinned here on
# purpose: a library should state what it needs, and the application pins
# live in requirements.txt / requirements-full.lock.
INSTALL_REQUIRES = [
    # --- core model stack ---------------------------------------------
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "trl",                    # SFTTrainer, used by Phase 2
    "datasets",
    "tokenizers",
    "sentencepiece",
    "safetensors",
    "huggingface_hub",

    # --- vision / OCR --------------------------------------------------
    "pillow",
    # Phase 3 and bhasha/eval/ocr_models.py. Core rather than optional:
    # the OCR pipeline is the paper's third contribution.
    "qwen-vl-utils",

    # --- serving -------------------------------------------------------
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",       # required by FastAPI for UploadFile on /ocr

    # --- configuration -------------------------------------------------
    "PyYAML",                 # bhasha/config.py reads configs/*.yaml

    # --- evaluation ----------------------------------------------------
    "jiwer",
    "sacrebleu",              # Sec. V-A BLEU / chrF++; see ERRATA B8

    # --- utilities -----------------------------------------------------
    "numpy",
    "pandas",
    "scikit-learn",
    "tqdm",
]

EXTRAS_REQUIRE = {
    # The legacy PaddleOCR + Tesseract pipeline in bhasha/ocr/pipeline.py,
    # and the optional `--detector paddle` stage of the hybrid pipeline.
    # Not part of the methodology in the paper -- docs/ERRATA.md group C.
    "paddle": ["paddleocr", "paddlepaddle", "opencv-python"],
    # Only bhasha/app/api.py's /api/chat and /api/philosophical, which make
    # outbound network calls and are not part of the paper's pipeline
    # (docs/ERRATA.md C3).
    "legacy-api": ["google-generativeai"],
    # scripts/audit_text_corpus.py, when no unar/unrar/7z is on PATH.
    "archive": ["rarfile"],
    "dev": ["pytest"],
}
EXTRAS_REQUIRE["all"] = sorted(
    {pkg for group in EXTRAS_REQUIRE.values() for pkg in group}
)

setup(
    name="bhasha",
    version="0.1.0",
    description=(
        "Bangla text generation and handwritten OCR on a single 16 GB GPU "
        "— artifact for the BhashaLLM paper"
    ),
    license="MIT",
    packages=find_packages(include=["bhasha", "bhasha.*"]),
    # Ship the non-Python files the package needs at runtime.
    package_data={"bhasha": ["app/static/*.html"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            # The Sec. III-F interface, importable after installation.
            "bhasha-models=test_models:main",
        ],
    },
)
