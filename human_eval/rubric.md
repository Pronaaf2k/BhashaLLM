# Translation and creative-writing rubric

Section III-E of the paper specifies a 5-point rubric across three
dimensions but does not define what separates adjacent points. Undefined
scale points are not reproducible even in principle, and they are the
usual source of low naturalness agreement. This file supplies the anchors.

Scored blind: model identity hidden, output order randomised. Record the
randomisation seed and the blinding map (`human_eval/blinding_map.json`) so
the blinding is verifiable after the fact.

Collect as one row per rating:
`rater_id,item_id,model_id,dimension,score`

Then: `python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv`

---

## Dimension 1 — Semantic accuracy

Does the output carry the meaning of the source?

| Score | Anchor |
| --- | --- |
| 1 | Meaning is unrelated to the source, or the output is empty/degenerate. |
| 2 | Topic is recognisable but the proposition is wrong; a reader would be misinformed. |
| 3 | Main clause is correct; one or more subordinate clauses, modifiers or named entities are lost or wrong. |
| 4 | Fully correct meaning with a minor loss of nuance, register or emphasis. |
| 5 | Complete and faithful, including nuance and implicature. |

## Dimension 2 — Script correctness

Is the output in correct Bangla script throughout?

| Score | Anchor |
| --- | --- |
| 1 | Predominantly not Bangla script (English or Devanagari). |
| 2 | Bangla with sustained passages in another script, or pervasive conjunct/matra corruption. |
| 3 | Bangla throughout with isolated foreign codepoints or several malformed conjuncts. |
| 4 | Correct Bangla with at most one or two orthographic slips (a wrong matra, a missing hasant). |
| 5 | Fully correct Bangla orthography; no foreign codepoints, all conjuncts well-formed. |

Note: score this dimension **after** the automatic check in
`eval/script_integrity.py`. The automatic detector settles the binary
question of foreign codepoints; the rater judges orthographic quality,
which the detector cannot see. Raters should not be shown the detector
output before scoring.

## Dimension 3 — Naturalness

Would a native speaker write this?

| Score | Anchor |
| --- | --- |
| 1 | Not parseable as Bangla prose. |
| 2 | Parseable but plainly machine-produced: calqued word order, English syntax in Bangla words. |
| 3 | Grammatical but stilted; a native speaker would rephrase most sentences. |
| 4 | Reads naturally with occasional awkward collocations or register slips. |
| 5 | Indistinguishable from competent native writing, register appropriate to the prompt. |

This dimension carries the most genuine subjectivity and will usually show
the lowest agreement. Report its alpha separately rather than pooling.

---

## Reporting requirements

- State **N items** and **N raters** as fixed numbers. "Two to three
  raters" is not reportable; the reader cannot tell what Table V rests on.
- Report scores as **means with standard deviation and n**, not as bare
  integers out of 5. Integers imply a single rating pass.
- Compute Krippendorff's alpha on the **ordinal** scale, **per dimension**.
  A 1-vs-5 disagreement is worse than a 3-vs-4 disagreement and nominal
  alpha cannot express that.
- **Report the alpha that comes out.** Section VI-D pre-commits to
  reporting it whatever its value. Following through on a pre-registered
  commitment is a credibility asset; re-running until it improves is not.
- If alpha lands in the moderate band (0.40–0.60), say so and say what it
  means: the model ranking is robust on the dimensions with higher
  agreement, and the naturalness comparison should be read as indicative.
