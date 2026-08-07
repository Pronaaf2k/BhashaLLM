#!/usr/bin/env bash
# Measure the local footprint claimed in paper Sec. III-F.
#
#     "The full local footprint is 10.5GB, split roughly 66% base models,
#      34% adapters, and under 1% code."
#
# docs/TRACEABILITY.md lists docs/footprint.txt as the evidence file for
# that sentence with status CHECK, and no such file existed. The three
# percentages are the kind of claim a reader can verify in ten seconds if
# the artifact is present and cannot verify at all if it is not.
#
# Usage:
#     bash scripts/capture_footprint.sh              # writes docs/footprint.txt
#     bash scripts/capture_footprint.sh /path/to/out

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/footprint.txt}"
mkdir -p "$(dirname "$OUT")"

# Bytes for a path, or 0 when absent. `du -sb` is GNU; macOS needs `du -sk`.
size_bytes() {
    local p="$1"
    [ -e "$p" ] || { echo 0; return; }
    if du -sb "$p" >/dev/null 2>&1; then
        du -sb "$p" | cut -f1
    else
        echo $(( $(du -sk "$p" | cut -f1) * 1024 ))
    fi
}

human() { awk -v b="$1" 'BEGIN{printf "%.2f GB", b/1e9}'; }
pct()   { awk -v a="$1" -v b="$2" 'BEGIN{ if (b>0) printf "%.1f%%", 100*a/b; else printf "n/a" }'; }

BASE=$(size_bytes "$ROOT/models/base_models")

ADAPTERS=0
for d in "$ROOT/models/bangla_adapters" "$ROOT/models/instruct_adapters" \
         "$ROOT/models/ocr_adapters" "$ROOT/models/adapters"; do
    ADAPTERS=$(( ADAPTERS + $(size_bytes "$d") ))
done

# Code = the repository minus models/, data/ and .git/. Those three are the
# only large exclusions, and lumping them into "code" is what would make the
# "under 1%" claim untrue.
TOTAL_REPO=$(size_bytes "$ROOT")
GIT=$(size_bytes "$ROOT/.git")
DATA=$(size_bytes "$ROOT/data")
MODELS=$(size_bytes "$ROOT/models")
CODE=$(( TOTAL_REPO - GIT - DATA - MODELS ))
[ "$CODE" -lt 0 ] && CODE=0

FOOTPRINT=$(( BASE + ADAPTERS + CODE ))

{
    echo "BhashaLLM local footprint"
    echo "Evidence for paper Sec. III-F: '10.5GB, split roughly 66% base"
    echo "models, 34% adapters, and under 1% code'."
    echo
    echo "Captured : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Root     : $ROOT"
    echo "Commit   : $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
    echo
    printf "%-22s %14s %10s   %s\n" "Component" "Size" "Share" "Sec. III-F claim"
    printf "%-22s %14s %10s   %s\n" "----------------------" "--------------" "----------" "----------------"
    printf "%-22s %14s %10s   %s\n" "Base models"  "$(human $BASE)"     "$(pct $BASE $FOOTPRINT)"     "~66%"
    printf "%-22s %14s %10s   %s\n" "Adapters"     "$(human $ADAPTERS)" "$(pct $ADAPTERS $FOOTPRINT)" "~34%"
    printf "%-22s %14s %10s   %s\n" "Code"         "$(human $CODE)"     "$(pct $CODE $FOOTPRINT)"     "<1%"
    printf "%-22s %14s %10s   %s\n" "TOTAL"        "$(human $FOOTPRINT)" "100%"                       "10.5 GB"
    echo
    echo "Excluded from the footprint (present but not part of the claim):"
    printf "  %-20s %14s\n" ".git"   "$(human $GIT)"
    printf "  %-20s %14s\n" "data/"  "$(human $DATA)"
    echo
    if [ "$BASE" -eq 0 ] && [ "$ADAPTERS" -eq 0 ]; then
        echo "NOTE: models/ is empty on this machine, so only the code figure"
        echo "is real. Base models and adapters total ~10.5 GB and are not in"
        echo "git; see README section 'Models'. Rerun where they are present."
    fi
    LLAMA=$(size_bytes "$ROOT/llama_cpp")
    if [ "$LLAMA" -gt 10000000 ]; then
        echo "NOTE on the '<1% code' claim: llama_cpp/ holds $(human $LLAMA) of"
        echo "compiled binaries and is counted as code above. If the code share"
        echo "prints above 1%, this is why. Sec. III-F's '<1%' is plausible for"
        echo "source alone and not for source plus vendored binaries; say which"
        echo "the figure refers to."
        echo
    fi
    echo "Per-directory detail:"
    for d in models/base_models models/bangla_adapters models/instruct_adapters \
             models/ocr_adapters bhasha eval configs tests docs scripts \
             benchmarks llama_cpp training report; do
        [ -e "$ROOT/$d" ] && printf "  %-32s %14s\n" "$d" "$(human "$(size_bytes "$ROOT/$d")")"
    done
} | tee "$OUT"

echo
echo "wrote $OUT"
