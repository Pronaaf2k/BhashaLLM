#!/usr/bin/env bash
# Capture the real runtime environment. Never type version numbers into a
# paper by hand: Sec. IV-A of the published manuscript names four versions,
# none of which match this repository, and two of which cannot run on the
# GPU the paper describes. See docs/ERRATA.md A1.
set -uo pipefail

OUT="${1:-docs/environment_capture.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "== BhashaLLM environment capture =="
  echo "captured: $(date -Iseconds)"
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
  echo "git status: $(git status --porcelain 2>/dev/null | wc -l) uncommitted changes"
  echo
  echo "-- system --"
  uname -a
  echo
  echo "-- gpu --"
  nvidia-smi 2>/dev/null || echo "nvidia-smi unavailable"
  echo
  echo "-- python --"
  python -c "import sys; print(sys.version)"
  echo
  echo "-- key packages --"
  python - <<'PY'
mods = ["torch", "transformers", "peft", "bitsandbytes", "accelerate",
        "trl", "datasets", "sacrebleu", "jiwer", "numpy"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:16s} {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{m:16s} NOT INSTALLED ({type(e).__name__})")

try:
    import torch
    print()
    print(f"torch.version.cuda        {torch.version.cuda}")
    print(f"torch.cuda.is_available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device name               {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability        sm_{cap[0]}{cap[1]}")
        props = torch.cuda.get_device_properties(0)
        print(f"total VRAM                {props.total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"torch device query failed: {e}")
PY
  echo
  echo "-- full freeze --"
  pip freeze
} | tee "$OUT"

echo
echo "wrote $OUT"
echo "Every version string in Sec. IV-A must appear in this file verbatim."
