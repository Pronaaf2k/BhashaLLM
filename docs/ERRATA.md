# Errata and clarifications

This document records every point where the accepted manuscript and this
repository do not agree, together with what is actually true in each case.

It exists because the paper is published and cannot be silently changed,
while the repository can. Rather than reshape the repository to imply
things the artifacts do not support, we list the divergences here. Every
item below is either a transcription error, an omission, or a claim that
needs a qualifier it did not receive in the manuscript.

**Manuscript.** BhashaLLM: A QLoRA-Based Framework for Bangla Text
Generation and Handwritten Character Recognition. S. A. Binaaf,
M. Y. Arafat, A. S. Chaklader, R. M. Rahman. Department of Electrical and
Computer Engineering, North South University.

**Status of this document.** Living. Last updated: see git history.

---

## Start here

Fifty entries follow. These five are the ones that change what the paper
can claim, in descending order of severity. Everything else is a
qualifier, a transcription error, or a repository defect.

| # | What | Bears on |
| --- | --- | --- |
| [**A6**](#a6-the-repositorys-own-ocr-reports-contradict-table-vii-and-disagree-with-each-other) | Table VII's 28% → 12% has no artifact, and two committed reports reverse its sign | Abstract, Table VII, Sec. V-B, Sec. VII |
| [**A7**](#a7-the-benchmarked-llama-32-11b-was-almost-certainly-a-3b-model) | The benchmarked "Llama-3.2-11B" was almost certainly Llama-3.2-**3B** | Abstract, Sec. I, III-B, Tables I/II/V/VI |
| [**A8**](#a8-phase-2-was-not-trained-on-grading-data) | Phase 2 trained on Kaggle spelling/morphology pairs, not grading data | Sec. IV-C, Table IV, ref [23] |
| [**A9**](#a9-all-three-phases-ran-fewer-steps-than-table-iv-states) | All three phases ran short: 417 / 135 / 675 against 500 / 500 / 2,000 | Table IV, Sec. IV-C |
| [**A10**](#a10-the-ocr-training-data-in-section-iv-c-is-wrong) | Phase 3 trained on BanglaWriting, which the paper never cites | Sec. IV-C, Table IV |

Two of the three training phases (A8, A10) are described in Section IV-C
with data that is not what the code reads. Verifying A7 takes one command
(`ollama list`) and A9 takes one file (`trainer_state.json` in each final
adapter).

Three entries record errors **in this document**, found by auditing it
against the repository: [C7](#c7-docserratamd-itself-contained-an-inaccurate-row)
and [C21](#c21-the-committed-ocr-data-cannot-be-loaded-on-any-other-machine)
each describe a group C row that claimed a fix which had not been made.

---

## How to read this

Items are grouped by what can still be done about them.

| Group | Meaning |
| --- | --- |
| **A** | Correctable in the camera-ready if that window is still open. Check your deadline first — these are cheap and worth the email to the chair. |
| **B** | The manuscript text stands as published. The correction lives here. |
| **C** | Repository-side defects. Fixed in this release; recorded for the reader who saw the earlier state. |

<!-- retired-ids-note:start -->
**On numbering.** A1–A5 are the original camera-ready items. A6–A10 were
added later and are the more severe group; they were briefly numbered
`A0`, `A00`, `A01`, `A02`, `A03`, which collided with A1–A3 and is why the
scheme now runs 1–10 with no leading zeros. If you have a reference to an
`A0x` identifier, the mapping is A01→A6, A00→A7, A02→A8, A03→A9, A0→A10.
<!-- retired-ids-note:end -->

---

## Group A — correct in camera-ready if the window is open

### A1. Software versions in Section IV-A are wrong

Section IV-A reports PyTorch 2.1.2, Transformers 4.37.2, BitsAndBytes
0.43.0 and PEFT 0.7.1. The environment this work actually ran in, as
pinned in the repository, is:

| Component | Section IV-A | Actual |
| --- | --- | --- |
| PyTorch | 2.1.2 | 2.10.0 |
| Transformers | 4.37.2 | 4.57.6 |
| BitsAndBytes | 0.43.0 | 0.49.1 |
| PEFT | 0.7.1 | 0.18.1 |
| CUDA runtime | not stated | 12.8.x |

This is not a cosmetic mismatch. The RTX 5070 Ti is a Blackwell card
(compute capability sm_120). PyTorch 2.1.2 predates Blackwell support and
cannot target that architecture; Transformers 4.37.2 predates the model
classes required to load Qwen-2.5, Llama-3.2-Vision or a Qwen3-VL-style
OCR model. The versions named in the manuscript describe an environment in
which this project could not have run. The versions in the right-hand
column describe one in which it could.

Correct figures are captured mechanically by
`scripts/capture_environment.sh` into `docs/environment_capture.txt`. That
file, not a hand-typed list, is the citable record.

### A2. Reference [7] has a transposed arXiv identifier

Reference [7] cites arXiv:2010.01192 for *An Image Is Worth 16x16 Words*.
The correct identifier is **arXiv:2010.11929**. The digits are transposed
and the cited identifier resolves to an unrelated record.

Worth fixing above its apparent size: a transposed identifier is the
standard signature of a reference that was never opened, and it is
routinely checked.

### A3. Table VII reports the less favourable of two available baselines

Table VII lists Maung et al.'s hybrid pipeline at 10.37% CER alongside
BhashaLLM at 12%. 10.37% is their **pre-correction** figure. Their final
reported system, after the Word2Vec spelling-correction stage, reaches
**2.47%** — a number Section II-A of our own paper already cites. The
manuscript therefore contains both figures while the comparison table
shows only the one favourable to us.

The table should carry both, with a footnote distinguishing them. The
comparison against Google Cloud Vision at 13.89% is unaffected and remains
the stronger and fairer point.

### A4. Abstract overstates the OCR training set

The abstract describes the OCR model as fine-tuned "on a self-collected
1,500-image dataset." Table IV gives the training split as 7,050 images:
6,000 sampled from Ekush plus 1,050 self-collected pages. The
self-collected material is roughly 15% of the training data, not the
whole of it. The 1,500-image figure is the size of the full self-collected
collection, of which 300 images form the held-out test split.

### A5. Abstract omits the writer-overlap qualifier

Section VI-D states plainly that the self-collected split is not
writer-disjoint and that 12% CER "is probably optimistic for a writer the
model has never seen." The abstract reports 12% next to Google Cloud
Vision's 13.89% with no such qualifier. A reader of the abstract alone
takes away a controlled comparison that Section VI-D withdraws.

---

---

## A6. The repository's own OCR reports contradict Table VII, and disagree with each other

Table VII reports that QLoRA fine-tuning improved handwritten OCR:

| Metric | Before | After |
| --- | --- | --- |
| Character accuracy | 72% | 88% |
| Character error rate | 28% | 12% |

The repository contains **three** OCR evaluations. None of them produces
those numbers, and one reverses the sign of the result.

**1. `report/ocr_benchmark_report.md`** — baseline vs fine-tuned
`swapnillo/Bangla-OCR-SFT`, the paper's own model, on Ekush:

| Model | Accuracy | CER | WER |
| --- | --- | --- | --- |
| Baseline | 66.00% | **18.10%** | 34.00% |
| Fine-tuned | 5.00% | **92.72%** | 98.00% |

The report's own summary line reads `+-61.00%` improvement. **Fine-tuning
made the model 61 accuracy points worse and raised CER from 18% to 93%.**
Table VII states the opposite direction and a different magnitude. That
report also gives the split as 900 training / 100 test images, where
Table IV gives 7,050 / 300.

**2. `report/COMPREHENSIVE_OCR_EVALUATION.md`** — three approaches on 50
Ekush images:

| Model | Accuracy |
| --- | --- |
| Baseline Swapnillo | **0.00%** |
| Fine-tuned Swapnillo | **0.00%** |
| PaddleOCR | not available |

Its sample predictions show the model answering the digit `৩` with `।`,
`'` and `T`. This is a total failure, not an 88% system.

**3. `training/vlm_ocr/outputs/paligemma2_3b_448_desktop_lines_lora/desktop_lines_test_predictions.jsonl`**
— 88 line predictions with references, the only committed OCR predictions
in the repository. Scored with this repository's own `eval/ocr_cer.py`:

| Metric | Value |
| --- | --- |
| CER | **52.69%** |
| Character accuracy | **47.31%** |
| Reference characters | 2,866 over 88 lines |

Per grapheme category: consonants 50.7%, vowel diacritics 58.6%,
consonant diacritics 53.6%, independent vowels 55.1%. Section V-B reports
91% / 85% / 79% / 94%, and in a different rank order — Section V-B has
consonants above vowel diacritics, and the measured data has the reverse.

**Summary of the four numbers.**

| Source | CER | Model |
| --- | --- | --- |
| Paper Table VII, after fine-tuning | 12% | Bangla-OCR-SFT + adapter |
| `report/ocr_benchmark_report.md`, baseline | 18.1% | Bangla-OCR-SFT |
| `report/ocr_benchmark_report.md`, fine-tuned | 92.7% | Bangla-OCR-SFT + adapter |
| PaliGemma predictions, measured here | 52.7% | PaliGemma-2 3B (not in the paper) |

**Position taken here.** The 28% → 12% improvement in Table VII, the
abstract, and Section VII's conclusion has **no supporting artifact in
this repository**, and the two committed evaluations of the paper's own
model both report that fine-tuning made it worse. This is the single
largest gap between the manuscript and its artifact, and unlike A7 it
cannot be resolved by relabelling.

**What to do.** Produce the predictions behind Table VII —
`python -m bhasha.ocr.hybrid_pipeline --manifest ... --split test --out
eval/ocr_predictions.jsonl`, then `python eval/ocr_cer.py`. If they cannot
be produced, Table VII must be withdrawn rather than qualified. If the two
existing reports are from a broken run, say so and delete or annotate
them; leaving three mutually contradictory OCR evaluations in the
repository is worse than any one of them being wrong.
`scripts/recover_committed_evidence.py` prints all four figures side by
side.

---

## A7. The benchmarked "Llama-3.2-11B" was almost certainly a 3B model

**This is the most serious item in this document. It bears on the paper's
central conclusion.**

`bhasha/llm/run_benchmark_suite.py` is the script that produced the
generations in `llm outputs/`. Every one of those files carries the header
"Benchmark Outputs (Ollama Backend)". The script's model table maps the
paper's model names onto Ollama tags:

```python
MODELS = {
    "Qwen_1.5B":     "qwen2.5:1.5b",
    "Gemma_2B":      "gemma2:2b",
    "Qwen_3B":       "qwen2.5:3b",
    "Mistral_7B":    "mistral:latest",
    "Gemma_9B":      "gemma2:9b",
    "Nemo_12B":      "mistral-nemo:latest",
    "Llama_3.1_8B":  "llama3.1:latest",
    "Llama_3.2_11B": "llama3.2:latest",      # <-- this line
    "bn_rag_8B":     "hf.co/BanglaLLM/bangla-llama-13b-base-v0.1-GGUF"
}
```

**`llama3.2:latest` is the 3-billion-parameter text model.** In Ollama's
library the `llama3.2` tag defaults to `llama3.2:3b` — 3.21B parameters,
Q4_K_M, a 2.0 GB download. Llama-3.2 was released in 1B and 3B text sizes;
the 11B model is *Llama-3.2-11B-Vision* and is served under a different
tag entirely (`llama3.2-vision:11b`).

Every other entry in the table is consistent with its label. The explicit
tags (`gemma2:2b`, `gemma2:9b`, `qwen2.5:1.5b`, `qwen2.5:3b`) name their
sizes, and `mistral:latest` (7B), `llama3.1:latest` (8B) and
`mistral-nemo:latest` (12B) resolve to the sizes Table I gives. **Only the
Llama-3.2 row is wrong, and it is the row the paper's conclusion rests
on.**

**What this affects.** Everything attributed to "Llama-3.2-11B":

| Claim | Location |
| --- | --- |
| "Llama-3.2-11B is identified as the strongest candidate" | Abstract |
| "identifying Llama-3.2-11B as the strongest candidate under a 16GB VRAM budget" | Sec. I |
| Selected as the primary generation and translation model | Sec. III-B |
| 14/15 on the human rubric, 5/5 semantic accuracy, 5/5 script correctness | Table V |
| Highest ROUGE of the set, 0.52 / 0.26 / 0.50 | Table VI |
| "over 99% of generations" in correct Bangla script | Abstract, Sec. V-C |
| 88% correct fixes on OCR correction | Sec. V-B |
| 1450 ms, 69 tokens/s | Sec. V-C |
| "an 11-billion-parameter model trainable on 16GB" | Abstract, Sec. VI-E |
| Table II's entire feasibility analysis | Table II, Sec. III-C |

Group B1 already records that no 11B model was ever *fine-tuned*, and took
the position that it was at least "evaluated by inference under
quantisation". This entry withdraws even that: the evidence in the
repository indicates the model evaluated was Llama-3.2-**3B**.

**Why this is recoverable.** The finding does not damage the result, it
relocates it. "A 3B model held Bangla script in over 99% of generations and
beat every larger model tested" is a *more* interesting claim than the one
in the paper, and it fits the hardware story better: a 3B model at Q4_K_M
is about 2 GB, which makes the 16 GB budget comfortable rather than tight.
The Section VI-B discussion of small models would need rewriting, and
Table II becomes a pure feasibility analysis with no model behind it.

**What to do.** Confirm by running `ollama list` on the machine that
produced `llm outputs/` and reading the digest and size for the
`llama3.2` entry. If it is 2.0 GB, the model was the 3B. Then either
re-run the benchmark with `llama3.2-vision:11b` and report those numbers,
or relabel throughout. Do not publish the 11B attribution without one or
the other.

---

## A8. Phase 2 was not trained on grading data

Section IV-C describes Phase 2:

> Instruction tuning for grading used the same base model with the Bangla
> adapter loaded. It trained on 400 synthetic and human-written
> instruction-grading pairs, with 100 pairs held out for validation, and
> converged to a validation loss of 0.018.

Reference [23] describes the source: *"Synthetic grading dataset
(self-generated), 200 question-answer pairs generated from Bangla
literature for grading logic training."*

**No such dataset exists in this repository, and the Phase-2 trainer reads
something else entirely.** `bhasha/llm/train_instruct.py` loads
`data/processed/kaggle_instruct.jsonl`, which is built by
`bhasha/llm/prepare_kaggle_data.py` from four Kaggle resources:

| Source | Ref | Task the script generates | Volume |
| --- | --- | --- | --- |
| Spelling checker (`right_file`/`wrong_file`) | [24] | "correct this misspelled word" | up to **20,000** pairs |
| Bangla morphological dataset | [26] | "is this sentence literal or metaphorical?" | all rows |
| 80k word frequency list | [25] | "is this a correct Bangla word?" | sampled |
| OSCAR Bengali sentences | [27] | "correct any language errors in this sentence" | sampled |

Four separate problems.

**1. None of these tasks is grading.** Spelling correction,
literal-versus-metaphorical classification, vocabulary lookup and
proofreading are four language tasks. Section III-B's requirement is to
"grade short written answers with Bangla feedback," and Table IV calls the
phase "Grading SFT". Nothing in the training data has the shape
*(question, reference answer, student answer) → feedback*.

**2. The volume is off by roughly two orders of magnitude.** The spelling
section alone takes up to 20,000 pairs. Table IV states 400 train / 100
validation / 50 held out.

**3. References [24]–[27] change role.** Section IV-B introduces them as
resources against which "tokenisation and morphology behaviour were
sanity-checked". They are the Phase-2 *training set*. A sanity-check
resource and a training set are different claims, and the second is the
one the artifact supports.

**4. One task trains a constant answer.** The OSCAR proofreading pairs are
generated as:

```python
q = f"নিচের বাক্যটিতে কোনো ভাষার ভুল থাকলে সংশোধন করুন: '{s}'"
a = f"বাক্যটি সঠিক আছে: '{s}'"
```

The question asks the model to correct any errors; the answer is always
"the sentence is correct", for every sentence, because the sentences are
drawn from a corpus and never corrupted. This teaches the model to answer
"correct" regardless of input. It is also a plausible contributor to the
0.018 validation loss in Table IV: a task whose answer is a fixed template
around the echoed input is close to trivial, and a validation loss two
orders of magnitude below the Phase-1 and Phase-3 losses is what that
would look like.

**This is the second training-data mismatch in the paper**, after A10
(Phase 3 trained on BanglaWriting rather than Ekush + self-collected).
Two of the three phases are described in Section IV-C with data that is
not what the code reads.

**Independent corroboration.** `.agent/workflows/train_evaluate_bangla_llm.md`,
the project's own training runbook, opens with: "Run the training script to
fine-tune the model using the prepared **`kaggle_instruct.jsonl`**
instruction dataset." The repository's operating procedure names the same
file the trainer reads. There is no step anywhere that produces or
consumes grading pairs.

**What to do.** Either produce the 500 grading pairs reference [23]
describes and retrain, or rewrite Section IV-C and Table IV to describe
the Kaggle-derived instruction set that was actually used — in which case
[24]–[27] move from "sanity-checked against" to "trained on", and the
0.018 loss needs re-examining against the constant-answer task.

The path for the first option already exists:
`configs/phase2_grading_sft.yaml` points at `data/grading/{train,val,test}.jsonl`
with the 400/100/50 split, and `train_instruct.py` uses those files when
they are present, falling back to the Kaggle file and printing a warning
when they are not.

---

## A9. All three phases ran fewer steps than Table IV states

Table IV's "Steps" column gives 500 / 500 / 2,000. The committed
checkpoint directories say otherwise. `docs/MODELS_SUMMARY.md` records the
saved checkpoint steps for each adapter:

| Phase | Table IV | Checkpoints saved | Last step | Shortfall |
| --- | --- | --- | --- | --- |
| 1. Bangla PT | 500 | 10, 100, 200, 300, 400, **417** | 417 | 17% |
| 2. Grading SFT | 500 | 20, 40, 60, 80, 100, 120, **135** | 135 | **73%** |
| 3. OCR SFT | 2,000 | 50, 600, **675** | 675 | **66%** |

**Phase 1 has two independent sources that agree.** The recovered training
history (C18) ends at step 417, and the last saved checkpoint is 417. That
agreement is what makes the other two rows credible: the same file records
all three, and where it can be cross-checked it is correct.

**Phase 2 is the largest discrepancy in the table.** 135 steps against a
stated 500. The checkpoint cadence corroborates it — `train_instruct.py`
sets `save_steps=20`, and the list is 20, 40, 60, 80, 100, 120 with 135 as
the final save, exactly the pattern a run ending at 135 produces. Table IV
also gives Phase 2 as 5 epochs; at effective batch 4, 135 steps is 540
sequences, which is 5 epochs only if the training set is about 108
examples — not the 400 the table states, and not the tens of thousands the
Kaggle file actually contains (A8).

**Phase 3 ran 675 steps, not 2,000.** Section IV-C presents the 2,000-step
figure as one of two deliberate deviations from Table III — "2,000
optimiser steps instead of 500, because the image set is an order of
magnitude larger than the text sets". That reasoning is stated for a run
that appears not to have happened at that length.

**Why this matters beyond the numbers.** Section IV-C's step counts are
presented as design decisions with justifications attached. If the runs
stopped at 417, 135 and 675, then either the runs were interrupted and the
table records intent rather than outcome, or the checkpoints are from
different runs than the ones reported. Both are recoverable positions; the
current one — a table of round numbers with no matching artifact — is not.

`python scripts/recover_committed_evidence.py` prints this comparison.

**What to do.** Read `trainer_state.json` inside each final adapter
directory; it carries `global_step` and the full log history. Correct
Table IV to the actual steps, or state that training was interrupted and
report the step reached.

---

## A10. The OCR training data in Section IV-C is wrong

**This was C1, the document's highest-priority open item. It is now
resolved, and the answer is the one that requires a correction to the
paper rather than a rename in the repository.**

Section IV-C states that OCR fine-tuning "applied the same QLoRA
configuration to the vision encoder's attention layers using the Ekush
dataset [28] plus the manually collected pages." Table IV gives the
Phase-3 training split as 7,050 images, footnoted as "6,000 Ekush images
plus 1,050 self-collected pages."

The committed adapter was trained on **BanglaWriting**, a separate public
handwritten Bangla dataset that the manuscript does not cite anywhere.

**Evidence, all of it in the repository:**

| File | Line | What it shows |
| --- | --- | --- |
| `bhasha/ocr/train.py` | `DEFAULT_DATA_DIR` | reads `data/processed/banglawriting` |
| `bhasha/ocr/train.py` | `DEFAULT_OUTPUT_DIR` | writes `models/ocr_adapters/banglawriting_adapter` |
| `bhasha/eval/ocr_models.py` | `adapter_path`, `test_data_path` | scores `banglawriting_adapter` against `data/processed/banglawriting/test.jsonl` |
| `bhasha/scripts/model_paths.py` | `"ocr"` | resolves the production OCR adapter to `banglawriting_adapter` |
| `bhasha/scripts/train_ocr_improved.py` | `data_sources` | combines Ekush **and** BanglaWriting, writing a separate `combined_ocr_adapter` |
| `bhasha/scripts/debug_dataset_shapes.py` | `datasets` | lists Ekush and BanglaWriting as distinct prepared directories |

The last two are what settle it. The repository can tell Ekush from
BanglaWriting — they are separate prepared directories and there is a
separate script that combines them into a differently-named adapter. The
adapter the evaluation code and the production path both point at is the
BanglaWriting one.

**Consequences.**

1. Section IV-C's description of the Phase-3 training data does not
   describe the run behind the headline 28% → 12% CER result.
2. BanglaWriting must be cited. It is absent from the reference list.
3. Table IV's "6,000 Ekush images plus 1,050 self-collected pages"
   footnote is unsupported by any committed artifact.
4. Every downstream statement that depends on the training composition —
   including the Ekush domain-gap discussion in B12 below — needs
   re-examination against BanglaWriting's actual composition.

**What was *not* changed.** The default paths in `bhasha/ocr/train.py` are
left pointing at BanglaWriting, so the committed adapter stays
reproducible. `configs/phase3_ocr_sft.yaml` describes the composition the
paper claims; running with `--config` trains that instead. Both are now
recorded in `logs/phase3_summary.json` on every run, via a
`training_data_note` field, so the provenance travels with the artifact.

**Recommended action.** If the camera-ready window is open, correct
Section IV-C and Table IV and add the BanglaWriting citation. If it has
closed, this entry is the correction. Do not describe the OCR training
data from the manuscript without pointing at this entry.

---

## Group B — manuscript stands; corrections recorded here

### B1. The 11B training claim

Four locations assert that an 11-billion-parameter model was trained:
the abstract ("make an 11-billion-parameter model trainable on 16 GB"),
Table II (QLoRA row marked "used here"), Section III-C ("peaked at 9.4 GB"),
and Section VI-E ("trained an 11B model without a single out-of-memory
error").

Section IV-C and Table IV describe exactly three training runs, and none
of them is an 11B model:

| Phase | Base model | Parameters |
| --- | --- | --- |
| 1. Bangla PT | Qwen-2.5-1.5B-Instruct | 1.5B |
| 2. Grading SFT | Qwen-2.5-1.5B-Instruct | 1.5B |
| 3. OCR SFT | swapnillo/Bangla-OCR-SFT | ~2-4B (VLM) |

The contributions list in Section I agrees with Table IV, describing "a
QLoRA fine-tuning pipeline that adapts a 1.5B-parameter model." The local
model store holds only Qwen2.5-1.5B-Instruct and Bangla-OCR-SFT, and all
three trained adapters are Qwen-derived.

**Position taken here.** Llama-3.2-11B was evaluated by *inference* under
quantisation, not fine-tuned. QLoRA fine-tuning was applied to the 1.5B
instruction model and the vision-language OCR model. Table II should be
read as a *feasibility analysis* of what would be required to fine-tune an
11B model on this card, not as a record of a run that was performed; the
9.4 GB figure is an estimate from that analysis unless and until a
training log is produced.

The accurate version of the claim is still a good claim, and is the one we
stand behind: **an 11B model can be served, and smaller models fine-tuned,
entirely within 16 GB.** If a Llama-3.2-11B QLoRA training log is later
recovered, it will be committed to `logs/` and this entry amended.

### B2. Phase-1 epoch count

Section IV-C states the corpus was packed into "about 2,578 training
sequences" and that 500 steps at effective batch 4 covers "roughly 0.8 of
an epoch." Table IV repeats 0.8.

These do not reconcile with the stated corpus size:

- 2,578 sequences × 512 tokens = 1.32M tokens, but Table IV gives the
  training split as 5.28M tokens.
- 5.28M ÷ 512 = 10,312 sequences. 500 steps × batch 4 = 2,000 sequences,
  which is **0.19 epochs**, not 0.8.

Either the packed corpus is roughly a quarter of the size Table IV states,
or Phase 1 covered about a fifth of an epoch. Phases 2 and 3 reconcile
correctly, which isolates the error to Phase 1.

The `epochs_covered` field is now computed rather than written by hand —
see `configs/phase1_bangla_pt.yaml` and the logging block in the README.
Whatever it prints is the correct figure. A run that covers 0.19 epochs
and says so is unremarkable; the reconciliation failure is the problem,
not the number.

### B3. Latency was measured at a 100-token budget, not 256

Section IV-C fixes text decoding at 256 new tokens. Section V-C reports
Llama-3.2-11B at 1450 ms and 69 tokens/s, and Qwen-1.5B at 680 ms and
147 tokens/s. Both products give **100 tokens**, not 256:

- 69 tok/s × 1.45 s ≈ 100
- 147 tok/s × 0.68 s ≈ 100

At 69 tokens/s, 256 tokens would take about 3.7 seconds. The latency
figures are correct for a 100-token generation budget and should be read
with that budget attached. The 256-token setting applies to the quality
evaluations, not the latency measurements.

### B4. The energy estimate uses a latency the paper contradicts

Section VI-F assumes "roughly 600 ms each" for Llama-3.2-11B inference and
derives approximately 42 Wh per 1,000 inferences. The arithmetic is
correct for 600 ms (250 W × 600 s = 41.7 Wh) but Section V-C reports
1450 ms for the same model — a factor of 2.4.

Recomputed at the reported 1450 ms: 250 W × 1450 s ≈ **101 Wh** per 1,000
inferences. The grid carbon intensity factor was also left unstated; the
20 g CO2 figure implies roughly 0.48 kg CO2/kWh, which is below published
figures for the Bangladesh grid. Any restatement should cite the factor
used.

### B5. Full-fine-tuning memory arithmetic

Section III-C gives ~21 GB weights, ~21 GB gradients, and "roughly four
times the weight size again for the AdamW moments and master copy,"
totalling "near 170 GB." 21 + 21 + (4 × 21) = 128 GB, not 170.

The total is defensible; the multiplier in the sentence is not. Reaching
~170 GB requires six times the fp16 weight size for optimiser state, which
is what fp32 first moment + fp32 second moment + fp32 master weights
actually costs (3 × 42.8 GB = 128 GB, plus 21 + 21 = 170 GB). The stated
total is right and the stated multiplier is wrong.

### B6. The perplexity comparison in Section III-B is not tokenizer-comparable

Section III-B selects Qwen-2.5-1.5B over XGLM-1.7B on per-token perplexity
(3.8 vs 73.15). Per-token perplexity is normalised by token count, and the
two models segment Bangla into different numbers of tokens for the same
string, so the two figures have different denominators and the ratio does
not carry the meaning attributed to it.

Separately, 3.8 on out-of-domain Bangla literary text is implausibly low
for a 1.5B multilingual model, and usually indicates heavy subword
fragmentation, evaluation on seen text, loss averaged over padding, or a
very small sample.

`eval/compute_bpc.py` recomputes the comparison as bits-per-character,
which is tokenizer-independent, with padding masked and the character
count reported. That is the defensible form of the argument. The
qualitative conclusion — that Qwen-2.5-1.5B is the better Bangla base
model at this scale — is supported independently by the script-integrity
and downstream task results, and does not rest on the perplexity figure.

### B7. Per-grapheme accuracies require an alignment step not described

Section V-B reports accuracy by grapheme category (vowels 94%, base
consonants 91%, vowel diacritics 85%, consonant diacritics 79%).
Section IV-B states that self-collected ground truth consists of **line
transcriptions** — strings, which carry no per-character labels.

Deriving per-category accuracy from strings requires edit-distance
alignment of hypothesis to reference, Unicode-category mapping of each
aligned character, and aggregation. None of that is described in the
manuscript. It is now implemented and committed as `eval/ocr_cer.py`, so
the breakdown is reproducible; figures regenerated with that script are
the ones to cite.

### B8. Metrics reported without a supporting library

Table VI reports ROUGE-1/2/L for nine models. The repository's dependency
file contains no ROUGE implementation — no `rouge-score`, no `evaluate`,
no `nltk` — and no `sacrebleu` for the BLEU and chrF++ figures described
as pending. Table VI could not be regenerated from a clean install of this
project as it stood.

`eval/text_metrics.py` now implements ROUGE-1/2/L directly, with the
tokenizer named in its own output. **Two tokenisation hazards were found
while writing it, both of which silently corrupt Bangla metrics:**

1. The google-research `rouge_score` default tokenizer applies
   `re.sub(r"[^a-z0-9]+", " ", text)`, which strips all non-ASCII and
   reduces any Bangla string to empty. It returns 0.0 for every Bangla
   pair.
2. A Unicode `\w`-based tokenizer is also wrong. Python's `\w` matches
   only what `str.isalnum()` accepts, and `isalnum()` is False for
   Unicode categories Mn and Mc — which is what Bangla vowel signs and
   the hasant are. Such a tokenizer splits every word at its first
   matra: `বাংলাদেশের` becomes `ব`, `ল`, `দ`, `শ`, `র`. In our tests this
   inflated ROUGE-1 and roughly halved ROUGE-2.

Any Bangla ROUGE figure published without naming its tokenizer should be
treated as unverified, including Table VI as printed.

### B9. Script-confusion rates had no stated definition

The 23% / under 1% / over 99% figures are the paper's central metric and
the manuscript gives no definition, detection method, or sample size.

`eval/script_integrity.py` now defines it: a generation is script-confused
if it contains at least one codepoint in the Devanagari block
(U+0900–U+097F), **excluding U+0964 and U+0965**. That exclusion is not a
detail. The danda and double danda live in the Devanagari block but are
shared Indic punctuation and are the correct sentence terminators in
Bangla; a detector without the exemption flags every correctly punctuated
Bangla sentence as script-confused. Any previously computed rate that did
not exempt them is an overestimate.

### B10. Quantisation of the benchmarked models is unrecorded

Gemma-2-9B, Mistral-Nemo-12B, Llama-3-8B, Llama-3.1-8B and Llama-3.2-11B
cannot be served in fp16 on 16 GB and were therefore quantised. The
manuscript does not report the format or bit width for any of them. The
presence of `llama_cpp/` indicates GGUF for at least some.

Quantisation materially affects both output quality and script integrity,
so this is a confound running through the whole nine-model benchmark.
`benchmarks/model_registry.json` is the intended home for the per-model
record; see the README.

### B11. Undefined quantities

| Quantity | Location | Status |
| --- | --- | --- |
| "Confidence score" 0.68 → 0.82 | Table VII | Not a defined output of an autoregressive VLM. Needs a definition (e.g. exponentiated mean per-token log-probability over the generated sequence) or withdrawal. |
| Correction / over-correction / false-positive rates | Sec. V-B | No denominators given. 88% of how many available errors, over how many items? |
| "Fifth most-used writing system", "171 conjunct forms", "250 million speakers" | Sec. I | Uncited. |
| Speculative BLEU range "8–20" | Sec. V-A | Speculation about an unmeasured quantity, placed in a results section. |

### B12. OCR domain gap between training and test

Ekush is a dataset of **isolated handwritten characters**; the
self-collected material is lines or pages of running text. Approximately
85% of the OCR training set is therefore isolated characters while 100% of
the test set is running text. The manuscript does not acknowledge this
gap, and it plausibly bears on the diacritic-placement errors reported in
Section V-B, since diacritic position is exactly what isolated-character
training under-specifies.

**Amended in light of A10.** This entry was written against the training
composition the manuscript describes. Since the committed adapter was in
fact trained on BanglaWriting, the 85% figure does not describe the run
behind the reported CER, and the domain gap has to be recomputed against
BanglaWriting's actual composition rather than Ekush's. The general point
— that an isolated-character training set under-specifies diacritic
placement in running text — survives; the number attached to it does not.

### B13. Single seed, no variance

Section IV-C states all runs used seed 42 and executed once, so reported
losses carry no variance estimate. This is correctly disclosed in the
manuscript and is repeated here only so that readers do not read
differences between the phase losses as significant.

---

## Group C — repository defects, fixed in this release

| Defect | Previous state | Now |
| --- | --- | --- |
| README addressed to the author | Emoji headers, second-person congratulation, summary of completed work | Rewritten as documentation for a reader reproducing the paper |
| Hardcoded absolute path | `/home/benaaf/Desktop/BhashaLLM_Export/...` in README | Removed; paths are relative |
| Documented scripts absent | README described `test_models.py`, `model_paths.py`, `download_base_models.py`; none present in the repository | README now documents what is actually present; see A6 note below |
| Entry point disagreement | Sec. III-F calls `test_models.py` "the primary production interface"; the repository entry point is `main.py` (FastAPI via uvicorn) | README documents `main.py`. Recorded here as a manuscript/repo divergence |
| No LICENSE | Absent — making the Section VII promise to release the handwriting dataset legally empty | MIT for code; CC BY 4.0 stated separately for data |
| No CITATION.cff | Absent | Added |
| No release or tag | `main` moves; a reader six months on sees different code than the paper describes | Tag `v1.0-paper` + archival DOI (see README) |
| Dependency file is a whole-machine freeze | 185 pinned packages including `chromadb`, `langchain`, `sentence-transformers` (a retrieval stack, next to Section III-D explaining that retrieval was rejected), `paddleocr`, `paddlepaddle`, `paddlepaddle-gpu`, `paddlex`, `pytesseract` (three OCR engines the methodology never mentions), plus `agentmail`, `posthog`, `kubernetes`, `modelscope` | Split: `requirements.txt` (direct dependencies) and `requirements-full.lock` (the freeze, preserved) |
| Conflicting pins | `paddlepaddle==3.3.0` and `paddlepaddle-gpu==2.6.2` pinned simultaneously | Neither is a pipeline dependency; both confined to the lock file |
| OCR adapter name contradicts the data description | `models/ocr_adapters/banglawriting_adapter` vs Section IV-C's Ekush + self-collected | **Resolved: the adapter was trained on BanglaWriting. The manuscript is wrong, not the directory name. See [A10](#a10-the-ocr-training-data-in-section-iv-c-is-wrong)** |
| Documented scripts absent (2) | `test_models.py` named in Sec. III-F as the primary production interface; not in the repository | Present at the repository root. See C2 |
| `bhasha.data` package absent | Imported by three modules; Phase-3 OCR training could not be imported at all | Added. See C2 |
| Committed configs read by nothing | `configs/phase*.yaml` carried Table III and detailed reconciliation notes; no code loaded them | `bhasha/config.py` makes the YAML authoritative. See C2 |
| Opaque data blob | `text dataset.rar`, no manifest, checksum, license or provenance | **This row previously claimed the archive had been replaced by extracted files and a checksum. It had not — see C7.** `scripts/audit_text_corpus.py` now extracts it, writes `data/text_raw.sha256`, and audits it against Table IV. Findings in C8 |
| `.agent/workflows` and "Note for AI Assistants" in `DIRECTORY_STRUCTURE.md` | Reads to a sceptical visitor as evidence of generated rather than built work | Moved to `docs/DIRECTORY_STRUCTURE.md` with the note removed and its four conventions preserved under a **Conventions** heading; a stub remains at the root so links do not break |
| Web frontend absent | Sec. III-F describes an optional web interface; no HTML, JS, template dir or static mount existed | Added, opt-in behind `BHASHA_ENABLE_UI=1`. See C6 |
| `environment.yml` unrunnable | CUDA 12.1 on a Blackwell card, no pins, Paddle as a core dependency | Corrected. See C10 |
| Orphaned outputs | `merged_ocr_llm_app/outputs/` holds four reports with no producing code | Retained and recorded. See C9 |

### C1. Which dataset trained the OCR adapter — RESOLVED

**Resolved. Promoted to [A10](#a10-the-ocr-training-data-in-section-iv-c-is-wrong)
at the top of this document.**

The Phase-3 loader was opened and read. The second of the two outcomes
this entry anticipated is the one that obtains: the committed adapter was
trained on BanglaWriting, not on Ekush plus self-collected pages, so
Section IV-C is wrong about the training data behind the headline OCR
result. BanglaWriting must be cited. The evidence table is in A10.

This entry is kept rather than deleted so that a reader who saw the
earlier state can follow what changed and why.

---

### C2. Components the paper describes that the repository did not contain

Each of these was named in the manuscript or in this repository's own
documentation, and no code implemented it. All are now present. Nothing
that previously existed was removed to make room for any of them.

| Component | Paper location | Previous state | Now |
| --- | --- | --- | --- |
| `bhasha.data` package | — | Imported by `bhasha/ocr/train.py`, `bhasha/scripts/train_ocr_improved.py` and `bhasha/scripts/debug_dataset_shapes.py`; **absent**, so Phase 3 raised `ModuleNotFoundError` on import | `bhasha/data/dataset.py` supplies `OCRDataset` and `collate_fn` |
| `--config` training | README, "Reproducing the paper" | `configs/phase*.yaml` were committed but no code read them; the documented command did not work | `bhasha/config.py`; all three trainers accept `--config` |
| Table III hyperparameters | Table III | Trainers hardcoded seven LoRA target modules, `paged_adamw_32bit`, no LR schedule, no seed, 1024-token context | Defaults are now Table III (`q_proj, v_proj`, AdamW, cosine after 50 warm-up steps, seed 42, 512 tokens). `--target-modules legacy` restores the previous set |
| Per-phase run summaries | README logging block; `docs/TRACEABILITY.md` | No code wrote `logs/phase*_summary.json`, so every row pointing at it was unbacked | `bhasha/utils/run_summary.py`; written by all three phases |
| `epochs_covered` computed | B2, `configs/phase1_bangla_pt.yaml` | Config instructed that it be computed; nothing computed it | Computed from `len(train_ds)` and logged |
| Sequence packing to 512 | Sec. IV-C | Phase 1 padded each line to full length instead of packing | `pack_sequences` in `bhasha/llm/train.py`, controlled by `data.packing` |
| NFKC + Latin stripping + 80/10/10 by document | Sec. IV-B, IV-C | No preprocessing code of any kind | `bhasha/data/text_corpus.py` |
| Ekush stratified sampling by grapheme root | Sec. IV-C | `ekush_images: 6000` recorded in the config; no sampler | `bhasha/data/ekush_sampling.py`, with a guaranteed floor per root so rare conjuncts actually survive |
| Handwriting manifest with `writer_id` | Sec. VII; `configs/phase3_ocr_sft.yaml` | Referenced by three traceability rows and by `eval/ocr_cer.py --group-by`; no schema or validator | `bhasha/data/manifest.py` |
| OCR-correction rates | Sec. V-B | Reported without denominators (B11) | `eval/ocr_correction.py`, every denominator reported |
| Confidence score | Table VII | Undefined quantity (B11) | `eval/confidence.py` defines it as `exp(mean log p)` and reports calibration error alongside |
| Latency at a stated token budget | Sec. V-C | No benchmark script; B3 records the missing budget | `eval/latency.py`, which requires `--max-new-tokens` |
| Energy / CO2 recompute | Sec. VI-F | B4 records the contradicted latency and unstated grid factor | `eval/energy.py`, which requires `--grid-factor` and `--grid-source` |
| Per-model quantisation record | B10 | `benchmarks/model_registry.json` named in the README; file absent | Present, with `FILL` markers for the fields only the authors can supply |
| `test_models.py` | Sec. III-F | Named as "the primary production interface"; absent from the repository | Present at the repository root; `main.py` unchanged |
| Single-resident-adapter runtime | Sec. III-F | The FastAPI app served a ResNet-34 classifier and proxied to the Gemini cloud API; no adapter was loaded or swapped | `bhasha/app/adapter_manager.py` + `bhasha/app/routes_v1.py`, mounted at `/api/v1` alongside the untouched legacy endpoints |

### C4. The LLM correction stage of contribution #3 was a placeholder

Section I lists as the third contribution:

> A hybrid handwritten Bangla OCR pipeline combining a fine-tuned
> vision-language recognizer with LLM-based error correction, reaching 88%
> character-level accuracy on a held-out, self-collected test set.

Section V-B reports that stage's performance in detail — 88% and 85%
correct fixes for the two Llama models, 35% for Mistral-7B against a 30%
over-correction rate.

`bhasha/ocr/pipeline.py` contained:

```python
def correct_text_with_llm(text, model_path=None):
    # Stub for LLM correction.
    # In real pipeline, load the finetuned model and prompt it.
    return text  # Placeholder
```

The correction stage was not implemented. It is the half of contribution #3
that distinguishes it from ordinary OCR, and every number in Section V-B's
correction paragraph is a measurement of it.

Two further divergences in the same file:

- **Its recognizer is not the paper's.** `bhasha/ocr/pipeline.py`
  recognises with a PaddleOCR + Tesseract ensemble. Neither engine appears
  anywhere in the manuscript. The paper's recognizer is the QLoRA-adapted
  Qwen-VL model of Section IV-C Phase 3.
- **Section III-A's detection/recognition split had no code.** The
  architecture section makes decoupling the central design claim
  ("allows the detector and the recognizer to be optimised and swapped
  independently") and no module implemented the two stages as separable
  components.

**Now.** `bhasha/ocr/hybrid_pipeline.py` implements the pipeline the paper
describes: a swappable detection stage (`projection` | `paddle` | `none`),
QLoRA Qwen-VL recognition through the adapter manager, and a real LLM
correction stage with guardrails against the aggressive-editing failure
Section IV-D warns about. `bhasha/ocr/pipeline.py` is **unchanged** apart
from a docstring pointing here; it remains a usable classical baseline.

The batch output of the hybrid pipeline writes `noisy` (pre-correction)
and `hypothesis` (post-correction) into one JSONL, so the recognizer and
the corrector can be scored separately — which is what Table VII and
Section V-B report separately.

### C5. The repository ships a retrieval stack the paper says was not used

Section III-D is a full subsection arguing that retrieval-augmented
generation was deliberately rejected:

> A retriever supplies more text to condition on without supplying anything
> the model lacks [...] neither failure mode this paper addresses is a
> knowledge failure.

The repository contains a working retrieval implementation:

| File | What it is |
| --- | --- |
| `bhasha/llm/rag_benchmark.py` | a RAG benchmark harness |
| `docs/PORAG_SETUP.md` | setup instructions for a retrieval pipeline |
| `llm outputs/bn_rag_8B.md` | generations from a RAG model |
| `requirements-full.lock` | `chromadb`, `chroma-hnswlib`, `langchain*`, `sentence-transformers` |

Group C above already noted the packages. The code and the recorded outputs
are the substantive part: a reader who opens `llm outputs/` finds a RAG
model's generations sitting beside the nine benchmarked models, in a
repository whose paper explains why retrieval was not used.

**Position taken here.** This is exploratory work that preceded or ran
alongside the reported experiments and did not feed any result in the
paper. Nothing in Tables V, VI or VII depends on it. That is a normal thing
for a research repository to contain, and it is only a problem when it is
undocumented — a reader cannot otherwise tell whether Section III-D
describes a decision or a description written after the fact.

**Nothing was removed.** The files are retained. Section VII lists
retrieval as the first item of future work, which makes an existing
harness an asset rather than a contradiction, provided it is labelled.

### C11. The benchmark ran on a serving stack the paper does not describe

Section IV-A specifies the software: "PyTorch 2.1.2, Transformers 4.37.2,
BitsAndBytes 0.43.0, and PEFT 0.7.1" (versions corrected in A1). Sections
III-C and III-F describe 4-bit NF4 quantisation via BitsAndBytes.

The nine-model benchmark used **none of that**.
`bhasha/llm/run_benchmark_suite.py` posts to `http://localhost:11434/api/generate` —
Ollama — and every file in `llm outputs/` is headed "Benchmark Outputs
(Ollama Backend)". Ollama serves **GGUF at Q4_K_M** by default, through
llama.cpp. The `llama_cpp/` directory in this repository, previously
unexplained, is consistent with this.

This is the answer to B10's open question about unrecorded quantisation,
and it is a better answer than "unknown": the benchmark quantisation was
Q4_K_M GGUF, not the NF4 the methodology describes. `benchmarks/model_registry.json`
now records this per model.

Two consequences worth stating plainly:

1. **The benchmarked models and the fine-tuned models were served
   differently.** Tables V, VI and VII compare Q4_K_M GGUF inference,
   while Phases 1–3 trained NF4 via BitsAndBytes. Quantisation format
   affects output quality and script integrity, so the benchmark does not
   measure the configuration the deployment section describes.
2. **`llama_cpp/` is part of the method, not a leftover.** It should be
   named in Section IV-A.

### C12. Table V and Table VI rest on one item per task

`run_benchmark_suite.py` defines exactly **four prompts** — one each for
Translation, Summarization, OCR Fix and Creative. There is no item set.

So the Table V scores (semantic accuracy, script correctness, naturalness,
each out of 5) are a rater's judgement of **a single translation**, and
Table VI's ROUGE-1/2/L are computed on **a single summary** per model.

This is why `docs/TRACEABILITY.md` could not find an evaluation item set:
N is 1. A ROUGE score on one sentence pair has no meaningful precision, and
a 14/15 against a 12/15 on one item is not a ranking. Section IV-C's
"Evaluation used fixed decoding so that comparisons across the nine models
are not confounded by sampling" addresses a much smaller source of variance
than the one that dominates here.

`benchmarks/items/` and the schema in `benchmarks/README.md` exist for the
fix: fix an item set of meaningful size, commit it, and re-run. Until then,
Tables V and VI should be read as illustrative examples rather than
measurements, and N should be stated.

### C13. The benchmark decoding parameters contradict Section IV-C

Section IV-C: "temperature 0.7, top-p 0.9, repetition penalty 1.1, and 256
new tokens for text."

`run_benchmark_suite.py`:

```python
"options": {
    "temperature": 0.3,   # using low temp for consistent tests
    "top_p": 0.9,
    "num_predict": 256
}
```

Temperature was **0.3, not 0.7**, and **no repetition penalty was set** —
Ollama's default `repeat_penalty` is 1.1, so that one may coincidentally
match, but it was not set explicitly and is not recorded. Only `top_p` and
the token count agree with the paper.

Temperature matters here more than usual: 0.3 suppresses exactly the
sampling excursions that produce the script drift the paper measures, so
the script-confusion rates were collected under settings that understate
the effect relative to the 0.7 the paper reports.

### C14. The nine benchmarked models are not Table I's nine models

Table I lists: Qwen-2.5-1.5B, Qwen-2.5-3B, Gemma-2-2B, Gemma-2-9B,
Mistral-7B-v0.3, Mistral-Nemo-12B, Llama-3-8B, Llama-3.1-8B,
Llama-3.2-11B.

`run_benchmark_suite.py` runs: Qwen_1.5B, Gemma_2B, Qwen_3B, Mistral_7B,
Gemma_9B, Nemo_12B, Llama_3.1_8B, Llama_3.2_11B, **bn_rag_8B**.

Two mismatches:

- **Llama-3-8B is not in the runner and has no output file.** There is no
  `Llama_3_8B.md` in `llm outputs/`. Yet Table V gives it 6/15 and Table VI
  gives it ROUGE 0.32 / 0.11 / 0.30 with output language "Bangla (poor)".
  Those numbers have no generation behind them in this repository.
- **`bn_rag_8B` is in the runner and has an output file**, and it is a
  retrieval-augmented model — the approach Section III-D explains was
  deliberately not used (C5). It is not in Table I.

`llm outputs/` also holds `Gemma_4_E2B.md` and `qwen_bangla.md`, neither of
which appears in Table I.

So the set of nine models in the paper and the set of nine models that were
run overlap in eight places and differ in one, and the differing one is a
model the methodology section rules out.

### C26. `pip install .` produced a package that could not run

`setup.py` and `requirements.txt` disagreed in both directions, and the
gaps were load-bearing.

**Missing from `install_requires`, needed at runtime:**

| Package | What breaks without it |
| --- | --- |
| `PyYAML` | `bhasha/config.py` imports `yaml` to read `configs/phase*.yaml`. Without it, **the entire Table III mechanism fails at import** and `python -m bhasha.llm.train --config ...` cannot start. |
| `trl` | `bhasha/llm/train_instruct.py` uses `SFTTrainer`. Phase 2 cannot run. |
| `python-multipart` | FastAPI raises on any `UploadFile` route, so `/api/v1/ocr` and `/api/analyze` both 500. |
| `sacrebleu` | Sec. V-A's BLEU and chrF++ (B8). |
| `accelerate`, `numpy`, `tokenizers`, `safetensors`, `sentencepiece`, `torchvision`, `tqdm`, `pydantic`, `huggingface_hub`, `jiwer` | assorted import failures |

**Declared as mandatory but not pipeline dependencies:** `paddleocr` and
`opencv-python` (the legacy pipeline — group C records that PaddleOCR is
not in the methodology), and `google-generativeai` (only the two legacy
endpoints that make cloud calls, C3). Every install pulled a second deep
learning framework and a cloud SDK.

**No `package_data` and no `MANIFEST.in`.** `find_packages()` carries
importable modules only, so neither `configs/*.yaml` nor
`bhasha/app/static/index.html` shipped. An installed copy could not read a
phase config and served 404 at `/ui`.

**Now.** `install_requires` mirrors `requirements.txt`; the optional
groups (`paddle`, `legacy-api`, `archive`, `dev`) hold what was
incorrectly mandatory, with the reason recorded against each;
`package_data` and `MANIFEST.in` ship the configs, the frontend, the
evaluation scripts, this errata and the traceability register. A
`bhasha-models` console script exposes the Sec. III-F CLI after
installation.

### C27. The API documentation describes a cloud deployment

`docs/API_DOCS.md` gave its base URL as
`https://your-app-name.onrender.com` and documented only the three legacy
endpoints. Two problems.

**It assumes a cloud host.** Sections III-F and VI-F build an argument on
the system being local and offline — it "lets students, teachers, and
small developers work with Bangla-language AI tools without depending on
cloud APIs or continuous network access". A deployment guide pointing at
Render is the **third** artifact in the repository that assumes network
access, after the Gemini endpoints (C3) and the retrieval stack (C5).

**It documented none of the paper's endpoints.** `/api/v1/*` and `/ui`
were absent, so the only documented surface was the one the paper does not
describe.

Corrected: both surfaces are documented, the local base URL is listed
first, and a table at the top states which endpoints are in the paper and
which are not.

### C25. `pytest` failed before running anything

`tests/` held seven files, all named `test_*.py`, and **none of them was a
test**. Between them they contained zero `assert` statements. Two broke
collection outright:

| File | Problem |
| --- | --- |
| `test_ocr_lang.py` | `def test_lang(lang_code)` — pytest reads the positional argument as a fixture request and errors with `fixture 'lang_code' not found` |
| `test_ocr_libs.py` | `def test_ocr_libraries(image_path)` — same |
| `test_eval.py` | `from evaluate_ocr_models import compute_metrics`; no such module exists anywhere |
| `test_models.py`, `test_image_processing.py` | `from model_paths import ...` rather than `from bhasha.scripts.model_paths import ...`, so they import only from inside `bhasha/scripts/` |
| `test_write.py` | one line: `# test` |
| `test_download_ocr.py` | downloads a dataset; no assertions |

`test_eval.py`'s missing import is the **third** module in this category,
after `bhasha.data` (C2) and `ekush_mapping` (C15).

The seven are useful diagnostics — they check whether PaddleOCR accepts
`lang='bn'`, whether EasyOCR and Tesseract are installed, whether the
adapters load. They are simply not tests, and the `test_` prefix made
pytest try to run them.

**Now.** `tests/conftest.py` excludes them from collection, with the reason
for each recorded in the file; `tests/README.md` documents what each one
does and how to run it directly. **Nothing was deleted.** `pytest tests/`
now runs the real suite — `test_paper_alignment.py`, 235 assertions across
78 tests, organised by the paper claim each group checks.

### C24. A planning document describes a framework the project never used

`docs/unsloth_tuning_notes.md` is written in the first person future —
"here is a comprehensive breakdown of why and how **we will use** Unsloth
to fine-tune our Qwen model" — and sets out an implementation flow built on
`unsloth.FastLanguageModel`.

Unsloth appears nowhere else: not in `requirements.txt`, not in
`requirements-full.lock`, not in any `.py` file. The training code uses
plain PEFT and Transformers, which is what Section III-F describes.

The document is a plan that was not followed, which is a normal thing for a
research repository to contain. It is recorded here only because a reader
who opens `docs/` finds a detailed argument for a framework the paper's
methodology section does not mention, and has no way to tell that it was
never adopted. A one-line header saying so would settle it.

`report/IMPLEMENTATION_PLAN.md`, `report/TASK_CHECKLIST.md` and
`report/WALKTHROUGH.md` are in the same category: planning and progress
documents, not results. `WALKTHROUGH.md` records Phase 1 starting at loss
1.54, where the recovered log (C18) shows 1.6046.

### C21. The committed OCR data cannot be loaded on any other machine

Every record in `training/vlm_ocr/data_desktop/` stores its image as an
absolute path on one developer's laptop:

```json
{"image": "/home/benaaf/Desktop/datasets/image_787 to image_927/img799line8.png", "text": "..."}
```

**3,402 of 3,402 records**, plus 88 more in the committed PaliGemma
predictions. On any other machine not one of them resolves. The
self-collected handwriting data — the material Section IV-B presents as
the project's own contribution, and which Section VII promises to release
— is unusable by anyone who is not the author.

**A third document makes the same claim.**
`docs/PATH_MIGRATION_SUMMARY.md` opens: "All hardcoded paths from the old
workspace directory have been updated to use relative paths based on the
current project structure." It lists the files it fixed — and none of the
dataset JSONLs is among them, because that migration was about a
*different* old root (`/home/node/.openclaw/workspace/BhashaLLM`). The
claim is true of what it covered and false as a general statement, and it
is the sentence a reader reaches for when asking whether the data is
portable.

**This is the second inaccurate row in group C.** That table claims:

| Defect | Previous state | Now |
| --- | --- | --- |
| Hardcoded absolute path | `/home/benaaf/Desktop/BhashaLLM_Export/...` in README | **This row was inaccurate: only the README instance was removed.** Absolute paths remain in nine Python files and all twelve dataset JSONLs, which makes the self-collected OCR data unloadable elsewhere. See C21; `scripts/relocate_dataset_paths.py` produces a portable copy |

Only the README instance was removed. Absolute paths remain in nine Python
files, two Markdown files, both `merged_ocr_llm_app` reports, and every one
of the twelve dataset JSONLs. C7 recorded the same failure for the corpus
archive; this is the same failure again, and the pattern is worth naming:
**a row in an errata that describes an intended fix rather than a
performed one is worse than no row.**

**Now.** `scripts/relocate_dataset_paths.py --write` produces a portable
copy under `training/vlm_ocr/data_portable/` with paths relative to a
declared root, keeping a `.pathmap.json` sidecar so the original layout is
not lost. `OCRDataset` and `bhasha/ocr/train.py` already accept
`--image_root`. The originals are untouched: the absolute paths are the
only surviving record of the collection folder structure, which is the
closest thing the repository has to the per-session provenance Section IV-B
describes.

### C22. The self-collected dataset does not have the shape Section IV-C gives

Measured with `scripts/relocate_dataset_paths.py`:

| | Paper Sec. IV-C | Committed data (`merged_lines`) |
| --- | --- | --- |
| Train | 1,050 | **1,529** |
| Val | 150 | **84** |
| Test | 300 | **88** |
| Total | 1,500 | **1,701** |
| Proportions | 70 / 10 / 20 | **90 / 5 / 5** |
| Unit | "pages" / "images" | **lines** |

Four problems.

**The unit is lines, not pages.** Filenames follow `img<N>line<M>.png`, and
the four split directories are named `*_lines`. The abstract's
"self-collected 1,500-image dataset", Section IV-B's "over 1,500
handwritten pages" and Table IV's "self-collected pages" all appear to be
counting line crops. 1,701 lines is the right order of magnitude for the
stated 1,500, which suggests the figure is real and the noun is wrong. A
page and a line are not interchangeable: 1,500 pages would be a
substantial corpus, and 1,701 lines is a small one.

**The split is 90/5/5, not 70/10/20.** A 5% test split of 88 lines is a
thin basis for the CER in Table VII, and it is not the split the paper
describes.

**Three collection directories are not three writers.** The data comes
from `image_787 to image_927` (2,650 records), `bnaf` (448) and
`dataset10thMay` (304). Section IV-B says three writers. A directory is a
collection session; nothing in the repository maps sessions to people, and
one session can contain several writers or one writer several sessions.

**No record carries a `writer_id`**, so the writer-disjoint evaluation
Section VI-D discusses cannot be computed from this data at all — not
"would show overlap", but cannot be computed. The manifest schema in
`bhasha/data/manifest.py` is where that belongs.

### C23. More uncited datasets, and a version the lockfile disagrees with

`training/vlm_ocr/DATASET_REPORT.md` describes a dataset collection the
paper never mentions:

| Dataset | Rows in the train split |
| --- | --- |
| `bangla-handwritten-datatset` (word-level) | 493,022 |
| `recognition_character_class` (BanglaLekha) | 149,340 |
| `bangla-handwriting-dataset-for-pix2pix` | 16,152 |
| `csai-hcr` (line/paragraph) | 895 |
| `bengali-handwritten-text-with-bounding-boxes` | 231 |

The manuscript cites only Ekush [28] and Bengali.AI graphemes [29].
Counting BanglaWriting (A10) and PaliGemma's desktop lines (C19), the
repository touches **at least seven** handwriting datasets, of which the
paper names two. None of these need to have been used for a reported
result, but `DATA_CARD.md` should say which were and which were not.

Separately, the committed PaliGemma adapter records
`"peft_version": "0.19.1"`, while `requirements.txt` pins `peft==0.18.1`.
A committed artifact already disagrees with the pinned environment. This is
minor next to A1, and it is the kind of thing
`scripts/capture_environment.sh` exists to catch.

### C18. Phase-1 training evidence exists, and corrects Table IV three times

`report/antigravity_experiments/logs/bangla_adapt_Qwen2.5-1.5B-Instruct_history.json`
is the HuggingFace log history for Phase 1. The filename is exactly the
`run_name` that `bhasha/llm/train.py` constructs, so this is that run.
`docs/TRACEABILITY.md` marked the Phase-1 rows `CHECK`; they can now be
resolved.

| Table IV, Phase 1 | Paper | Recovered log |
| --- | --- | --- |
| Steps | 500 | **410** |
| Final loss | 1.31, labelled *(train)* | train **1.3405**; **eval 1.3139** |
| Epochs | 0.8 | **0.985** |

Three corrections:

1. **The run stopped at 410 optimiser steps**, not 500.
2. **1.31 is the validation loss, not the training loss.** The final
   evaluation loss is 1.3139, which rounds to 1.31; the final training
   loss is 1.3405, which rounds to 1.34. Table IV's "Final loss" column
   labels the Phase-1 entry `(train)`. The number is right and the label
   is wrong.
3. **Epoch coverage was 0.985, essentially one full epoch**, not 0.8. The
   trailing summary entry records `"epoch": 1.0` and
   `"train_runtime": 582.44` seconds — under ten minutes, against
   Table IV's "3h05".

**This also resolves B2.** That entry could not reconcile 2,578 sequences
with a 5.28M-token training split, and computed 0.19 epochs from the
latter. The log settles it in favour of the smaller figure:
`train_samples_per_second` 2.859 × 582.44 s ≈ **1,665 sequences**, and
410 steps × effective batch 4 = 1,640 — one epoch, as logged. At 512
tokens per packed sequence that is roughly **0.85M tokens**, not the
5.28M Table IV gives for the training split.

That is independently consistent with C8, where the committed corpus
archive turned out to be far smaller than 6.6M tokens. Two separate lines
of evidence now point at the same conclusion: **the corpus is roughly an
order of magnitude smaller than Table IV states.**

`scripts/recover_committed_evidence.py` converts this history into
`logs/phase1_summary.json` in the schema `bhasha/utils/run_summary.py`
writes, so the Phase-1 row finally has an evidence file.

Phases 2 and 3 have no equivalent log. `report/training_metrics.csv`
matches neither (C16).

### C19. A third OCR model is committed and never mentioned

`training/vlm_ocr/` contains a complete, trained
**PaliGemma-2 3B (448px)** LoRA adapter — `final_adapter/adapter_model.safetensors`,
`checkpoint-576`, a processor, a dataset report, and 88 scored test
predictions. PaliGemma appears nowhere in the manuscript. The paper's OCR
model is `swapnillo/Bangla-OCR-SFT`, a Qwen3-VL-style architecture.

This is the **only OCR model in the repository with committed weights and
committed predictions**, which puts it in an odd position: the best-evidenced
OCR work in the artifact is work the paper does not describe.

It is also trained on a fourth dataset. `training/vlm_ocr/data_desktop/`
holds `bnaf_lines`, `dataset10thMay_lines`, `image_787_927_lines` and
`merged_lines` — line-level splits from `/home/benaaf/Desktop/datasets/`,
which is plausibly the self-collected material of Section IV-B but is
never connected to it in writing.

Nothing is removed. Either describe this work in the paper, or state in
`DATA_CARD.md` that it is exploratory and separate from the reported
results, as was done for the retrieval stack in C5.

### C20. A results file is committed empty

`report/antigravity_experiments/all_results_raw.json` is **0 bytes**. A
file named `all_results_raw.json` that contains nothing reads, to anyone
auditing the repository, as results that were meant to be there and are
not. Either populate it or remove it.

`merged_ocr_llm_app/outputs/` (C9) is a milder version of the same
problem: its two reports record `"images_tested": 1` with
`"ocr_exact_match": 0` and `"llm_exact_match": 0`. One image, no matches.

### C17. The OCR-correction prompt contains its own answer

The prompt sent to every model for the OCR-Fix task, verbatim from
`bhasha/llm/run_benchmark_suite.py`:

> Fix the spelling and grammatical errors in this broken OCR output. Do not
> add extra commentary:
>
> `'আিম বংলাদশ এ থািক। আমার দশনর নাম বংলাদশ। আমরা সবই ভই ভাই।' > (Expected: আমি বাংলাদেশে থাকি। আমার দেশের নাম বাংলাদেশ। আমরা সবাই ভাই ভাই।)`

**The correct answer is in the prompt.** A model does not have to correct
anything; it has to copy the string after `Expected:`. The task measures
instruction-following and copying, not Bangla orthographic repair.

This was found by converting the committed generations to JSONL and
scoring them with `eval/ocr_correction.py`
(`scripts/convert_llm_outputs.py`). Eight of the ten models score a
**correction rate of 1.00**, which is what copying produces and not what a
distribution of genuine correction ability looks like.

The three-rate design of Sec. IV-D is what makes the leak visible rather
than flattering. Correction rate alone says every model is perfect. Net
error reduction says otherwise:

| Model | correction rate | over-correction | net error reduction |
| --- | --- | --- | --- |
| Gemma-9B | 1.00 | 0.00 | **+1.00** |
| Llama-3.1-8B | 1.00 | 0.00 | **+1.00** |
| Llama-3.2-11B* | 1.00 | 0.00 | +0.86 |
| Qwen-3B | 1.00 | 0.02 | +0.93 |
| Nemo-12B | 1.00 | 0.00 | **−1.21** |
| Qwen-1.5B | 0.86 | 0.00 | **−6.14** |
| Mistral-7B | 0.00 | 0.80 | **−9.86** |
| bn_rag_8B | 1.00 | 0.22 | **−46.4** |

\* see A7 — this was probably the 3B model.

A negative net error reduction means the output is further from the
reference than the input was: the model copied the expected string *and*
appended commentary, despite "Do not add extra commentary". Sec. V-B's
reported 88% / 85% / 35% cannot be reproduced from these generations, and
the ordering it describes is only partly visible — Mistral-7B is indeed
the worst, which is the one clear agreement.

**What to do.** Rewrite the prompt without the `(Expected: ...)` clause,
keep the reference in the item file where the model cannot see it
(`benchmarks/items/ocr_correction.jsonl`, schema in
`benchmarks/README.md`), and re-run. Until then, Sec. V-B's correction
rates should be withdrawn rather than qualified: there is no
interpretation under which a leaked answer measures correction ability.

### C15. Two evaluation modules import a module that does not exist

`bhasha/eval/all_ocr.py` and `bhasha/eval/ocr_models.py` both begin with:

```python
from ekush_mapping import get_label_text
```

There is no `ekush_mapping.py` anywhere in the repository. Both modules
raise `ModuleNotFoundError` on import, so neither the comprehensive OCR
evaluation nor the baseline-vs-fine-tuned comparison could be run from a
clean checkout. This is the same class of defect as the missing
`bhasha.data` package (C2).

Two further scripts (`bhasha/scripts/debug_dataset_shapes.py`,
`bhasha/scripts/train_ocr_improved.py`) use `from model_paths import ...`
rather than `from bhasha.scripts.model_paths import ...`, so they import
only when the working directory happens to be `bhasha/scripts/`.

**Now.** `bhasha/eval/ekush_mapping.py` supplies `get_label_text` with the
Ekush label convention documented, and falls back to a path-derived label
when no mapping file is present, so the two modules import and run.

### C16. The only committed training log is of a crashed run

`logs/training_log.txt` is the repository's sole training log. It ends:

```
File ".../transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 281, in forward
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
RuntimeError: shape '[-1, 1280]' is invalid for input of size 262144
  0%|          | 0/1500 [00:00<?, ?it/s]
```

The run failed at step 0 of 1500 with a vision-tower shape mismatch. It is
not evidence for any row of Table IV.

Separately, `report/training_metrics.csv` holds 132 real training points
and 65 evaluation points, but reconciles with **no** row of Table IV: it
ends at train loss 0.2814, eval loss 0.2874, epoch 0.1356, at learning rate
2e-4. Table IV gives Phase 1 train 1.31, Phase 2 val 0.018, Phase 3 val
0.31 — and Phase 3 used 1e-4, so the learning rate rules that phase out
while the loss rules out the other two.

The three losses in Table IV currently have no artifact behind them.
`logs/phase{1,2,3}_summary.json`, written by `bhasha/utils/run_summary.py`,
is where they should come from.

### C6. The web frontend described in Section III-F did not exist

Section III-F states:

> A web frontend was added later as an optional interface for users who
> prefer not to use a terminal; it talks to the same backend and is not
> loaded unless explicitly opened, so it does not compete with the language
> models for GPU memory.

The repository contained no frontend of any kind: no HTML, no JavaScript,
no template directory, no static-file mount, no `package.json`, and no
Streamlit or Gradio entry point. The sentence had no artifact behind it.

**Now.** `bhasha/app/static/index.html` plus `bhasha/app/routes_ui.py`.
Each clause of the sentence is implemented rather than approximated: the
page calls only `/api/v1/*` so there is no second inference path ("talks to
the same backend"); the route is mounted **only** when `BHASHA_ENABLE_UI=1`
so a default `python main.py` serves the API alone ("not loaded unless
explicitly opened"); and it is one static file with no build step, npm, CDN
or framework ("does not compete with the language models for GPU memory").

### C7. `docs/ERRATA.md` itself contained an inaccurate row

The group C table below claimed:

| Defect | Previous state | Now |
| --- | --- | --- |
| Opaque data blob | `text dataset.rar`, no manifest, checksum, license or provenance | Replaced by extracted files + `data/text_raw.sha256` |

**That did not happen.** `text dataset.rar` was still the only form the
corpus existed in, no code extracted it, and `data/text_raw.sha256` was
never produced. An errata that asserts a fix which was not made is worse
than no errata, because it is the document a reader turns to precisely when
they have stopped trusting the paper.

The row has been corrected below, and `scripts/audit_text_corpus.py` now
makes the claim true when run: it extracts the archive, writes
`data/text_raw.sha256`, counts tokens, and compares the result to Table IV.

### C8. What the corpus archive actually contains

Auditing `text dataset.rar` surfaced three problems that bear directly on
Section IV-B and Section IV-C.

**The archive is 58 KB compressed.** Table IV states a 6.6M-token corpus.
Bangla compresses well, but 6.6M tokens is on the order of 25–30 MB of
UTF-8, which does not fit in 58 KB at any plausible ratio. Either the
committed archive is a sample rather than the training corpus, or the 6.6M
figure needs revising. Run `python scripts/audit_text_corpus.py
--tokenizer Qwen/Qwen2.5-1.5B-Instruct` for the measured count; whatever it
prints is the figure to cite. **The corpus behind Section IV-B is not
currently reproducible from this repository.**

**There is no author metadata at all.** The archive's 110 entries are named
`text dataset/1.txt` … `N.txt`, with no author directories and no
accompanying metadata. Section IV-C claims:

> The 6.6M-token Bangla corpus was split 80/10/10 by document rather than
> by sentence, so no work by the same author appears on both sides.

Nothing in the committed corpus records which author wrote which file, so
that guarantee cannot be reconstructed, verified, or reproduced.
`bhasha/data/text_corpus.py` handles this honestly rather than silently: it
groups by author when author information is present, and when it is not, it
reports in `split_manifest.json` that the author-disjointness claim **does
not hold** for the input it was given. Either add an author column and
re-split, or withdraw the claim.

**References [21] and [22] are not separable.** Two corpora are cited —
Kazi Nazrul Islam and Rabindranath Tagore — but the archive has a single
top-level directory. `DATA_CARD.md`'s per-source token counts cannot be
filled in from this file.

### C9. Orphaned evaluation outputs

`merged_ocr_llm_app/outputs/` contains four files —
`test_report_20260417_165123.{csv,json}` and
`test_report_20260417_165130.{csv,json}` — and there is no
`merged_ocr_llm_app` application anywhere in the repository. No script
produces these reports and nothing reads them.

They are retained; deleting evidence is worse than leaving it unexplained.
But a reader cannot trace them to a command, so they should not be cited,
and no figure in the paper depends on them. If they are the output of a
script that lives outside this repository, committing that script would
convert them from artifacts into evidence.

### C10. `environment.yml` described an environment that could not run

A third dependency file, disagreeing with both `requirements.txt` and the
captured environment:

- `pytorch-cuda=12.1`, which cannot target the RTX 5070 Ti. Blackwell is
  compute capability sm_120 and needs CUDA 12.8 or later. This is the same
  defect as A1, in a second file.
- No version pins on the pip block at all, so `conda env create` produced a
  different environment on every run and none of them matched
  `requirements.txt`.
- `paddlepaddle-gpu` and `paddleocr` as core dependencies, although group C
  below records that PaddleOCR is not part of the methodology in the paper.

Corrected: CUDA 12.8, pins matching `requirements.txt`, and the Paddle
packages moved to a commented optional block with the reason attached.

### C3. The shipped API contradicts the offline-deployment claim

Sections III-F and VI-F describe a deployment that runs entirely on one
local GPU, and Section VI-F builds an argument on it: local deployment
"lets students, teachers, and small developers work with Bangla-language
AI tools without depending on cloud APIs or continuous network access."

`bhasha/app/api.py` as committed calls `google.generativeai` from
`/api/chat` and `/api/philosophical`, and its `/api/analyze` endpoint
serves a ResNet-34 three-head grapheme classifier — an architecture that
appears nowhere in the paper and is not any of the three QLoRA adapters.

Those endpoints have **not** been removed; they are working functionality
and remain mounted. The paper-faithful surface is added beside them at
`/api/v1`, is fully local, and reports `"offline": true` from
`/api/v1/status`. A reader reproducing Section III-F should use `/api/v1`
or `test_models.py`. A reader benchmarking the shipped API should know
that two of its endpoints make outbound network calls.

---

## Standing commitment

Where a figure in the manuscript cannot be regenerated from this
repository, the honest options are to regenerate it, to report it with the
gap stated, or to withdraw it. Adjusting a number so that it reconciles
with another number is not among them.

Corrections to this document are welcome via the issue tracker.
