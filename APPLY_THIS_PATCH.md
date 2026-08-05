# How to apply this

Files to drop into `github.com/Pronaaf2k/BhashaLLM`, in the order that
makes each step verifiable.

Nothing here fabricates an artifact. The scripts produce evidence when you
run them; the documents record what is true now. Where the paper and the
repository disagree, `docs/ERRATA.md` says so rather than reshaping the
repository to imply otherwise — that is the version that survives someone
running `pip freeze`.

---

## Step 0 — Freeze the current state (10 minutes)

Before changing anything, preserve what the paper was written against.
Once you start editing you lose the ability to tell what was original.

```bash
git checkout -b archive/pre-errata-2026-08
git push -u origin archive/pre-errata-2026-08
git checkout main
```

Then inventory what evidence you already hold. **Do this before running
anything expensive** — `logs/`, `report/` and `llm outputs/` already exist
in your repository, and if the raw outputs behind Tables V, VI and VII are
sitting in them, most of the work below collapses from "regenerate" to
"point at it."

```bash
ls -laR logs/ report/ "llm outputs/" > EXISTING_OUTPUTS.txt
find . -type f -not -path "./.git/*" -printf "%p\t%s\t%TY-%Tm-%Td\n" \
  | sort > INVENTORY.tsv
```

Read `EXISTING_OUTPUTS.txt` against `docs/TRACEABILITY.md` and flip every
`CHECK` row to either `TRACED` or `MISSING`.

## Step 1 — Answer the one question that changes the errata (30 minutes)

Open the Phase-3 OCR training script and read which dataset loader it
calls. The committed adapter is named `banglawriting_adapter`; the paper
says Ekush plus self-collected pages. BanglaWriting is a different public
dataset the paper never cites.

```bash
grep -rn "banglawriting\|BanglaWriting\|ekush\|Ekush" --include="*.py" .
grep -rn "load_dataset\|ImageFolder\|read_csv" --include="*.py" training/ bhasha/ocr/
```

- **Legacy name, trained on Ekush + self-collected** → rename the
  directory, delete §C1 from `docs/ERRATA.md`, move on.
- **Actually trained on BanglaWriting** → Section IV-C is wrong about the
  training data behind your headline OCR result. Cite BanglaWriting, keep
  §C1, and promote it to the top of the errata.

Do not skip this. It is the one open item that changes what the errata has
to say, and it takes half an hour.

## Step 2 — Capture the real environment (15 minutes)

```bash
cp -r scripts/ configs/ eval/ human_eval/ docs/ /path/to/BhashaLLM/
bash scripts/capture_environment.sh
git add docs/environment_capture.txt scripts/capture_environment.sh
git commit -m "Capture actual runtime environment (torch 2.10, transformers 4.57.6, CUDA 12.8)"
```

Split the dependency file — keep the freeze, add a readable direct-deps
list:

```bash
git mv requirements.txt requirements-full.lock
cp /path/to/patch/requirements.txt requirements.txt
pip install sacrebleu          # needed for Sec. V-A; was never installed
git add requirements.txt requirements-full.lock
git commit -m "Split direct dependencies from the frozen environment"
```

## Step 3 — Land the documents (20 minutes)

```bash
cp README.md LICENSE CITATION.cff DATA_CARD.md /path/to/BhashaLLM/
cp docs/ERRATA.md docs/TRACEABILITY.md /path/to/BhashaLLM/docs/
```

Then, by hand:

- `CITATION.cff` — fill the venue, year and DOI from your acceptance notice.
- `README.md` — same, in the citation block near the top.
- `DATA_CARD.md` — every **`FILL`** marker. These are yours to answer; I
  left them empty rather than guessing, because a data card with invented
  numbers is worse than no data card.
- Move `DIRECTORY_STRUCTURE.md` into `docs/` and delete its "Note for AI
  Assistants" section. Combined with `.agent/workflows`, that section is
  read by a sceptical visitor as evidence the project was generated rather
  than built. The inference is unfair; the cue is removable.

```bash
git add -A && git commit -m "Add errata, traceability register, data card, license, citation"
```

## Step 4 — Regenerate what you can (variable)

Each command writes a JSON file that a reader can open. Run the cheap ones
first — script integrity and OCR CER need no retraining, only the
generations you already have.

```bash
# cheapest: needs only existing generations
python eval/script_integrity.py "llm outputs"/*.jsonl --out eval/script_integrity.json

# OCR: needs predictions + a manifest carrying writer_id
python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl \
    --manifest data/handwriting/manifest.csv --group-by writer_id \
    --out eval/ocr_cer.json

# ROUGE with a named Bangla tokenizer
python eval/text_metrics.py --pred benchmarks/raw/llama-3.2-11b.jsonl \
    --out eval/rouge_results.json

# base model selection, tokenizer-independent
python eval/compute_bpc.py --models Qwen/Qwen2.5-1.5B-Instruct facebook/xglm-1.7b \
    --corpus data/splits/test.txt --load-4bit --out eval/bpc_comparison.json

# human evaluation, once ratings.csv exists
python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv \
    --out human_eval/alpha_by_dimension.json
```

Commit every output. They are a few megabytes of JSON and CSV in total,
which is precisely why there is no reason not to have them in the repo.

## Step 5 — Release (1 hour)

The highest-value hour in this whole document.

```bash
git tag -a v1.0-paper -m "Version described in the BhashaLLM manuscript"
git push --tags
```

Then, on GitHub:

- Set the repository description and topics: `bangla`, `ocr`, `qlora`,
  `low-resource-nlp`, `vision-language-models`.
- Create a release from the tag.
- Connect the repository to Zenodo and archive the release to mint a DOI.
- Put the DOI in `README.md` and `CITATION.cff`.
- Publish the three adapters (3.6 GB total) on Hugging Face. Too large for
  git, free to host, and they are the direct artifact of the work.

A DOI is a timestamped, third-party-hosted record that this code existed
in this form on this date. It converts "trust us" into "check for
yourself," which is the entire point.

---

## One thing that is time-critical

Ask the proceedings chair whether the camera-ready window is still open.
Four items in `docs/ERRATA.md` group A are one-line fixes that are far
better made in the paper than in an errata:

1. **Reference [7]** — arXiv:2010.01192 should be **2010.11929**. A
   transposed identifier is the classic signature of a reference nobody
   opened, and it is routinely checked.
2. **Section IV-A versions** — replace with the captured environment.
3. **Table VII** — add Maung et al.'s final 2.47% alongside the
   pre-correction 10.37%. Your own Section II-A already cites 2.47%, so
   the paper currently contains both figures while the table shows only
   the flattering one. A reader who checks the reference reads that as
   cherry-picking even though it was an oversight.
4. **Abstract** — "self-collected 1,500-image dataset" should reflect
   Table IV's 7,050 training images, and the 12% CER needs the
   writer-overlap qualifier that Section VI-D already states.

If the window has closed, all four live in the errata instead, which is a
normal and respectable place for them to live. But it costs one email to
find out, and the camera-ready is the cheaper venue.

---

## What is deliberately not here

No fabricated training log for an 11B run that has no checkpoint. No
`environment_capture.txt` asserting PyTorch 2.1.2 on a Blackwell card. No
back-filled evaluation outputs matching the printed numbers.

Those would make the repository agree with the paper, and the agreement
would survive exactly until someone ran `pip freeze` or looked for the
checkpoint. A published paper with a maintained errata is ordinary
scholarship. A repository built to corroborate claims its artifacts
contradict is a different category of problem, and it is not recoverable.

The work behind this paper is real — a nine-model Bangla benchmark, a
three-phase QLoRA pipeline, a self-collected handwriting set, and a
working offline deployment on one consumer GPU is a substantial amount for
a student team. The evidence for it is on a laptop instead of in the
repository. Moving it is the whole job.
