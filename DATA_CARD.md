# Data card

Provenance, licensing and collection procedure for every dataset used.

Fields marked **`FILL`** are known only to the authors and must be
completed before the dataset is released. They are left explicit rather
than guessed, because a data card with invented numbers is worse than none.

---

## 1. Self-collected handwritten Bangla set

The central novel asset of this work, and the one the paper describes
least. Section VII commits to releasing it.

### Scale

| Field | Value |
| --- | --- |
| Units collected | 1,500 |
| Unit type | **`FILL`** — pages, or line images? The paper uses "pages" and "images" interchangeably. If pages were segmented into lines, give both counts. |
| Writers | 3 |
| Pages per writer | ~500 |
| Total word count | **`FILL`** |
| Total character count | **`FILL`** |
| Sheet size | **`FILL`** |
| Approximate words per page | **`FILL`** |
| Time to complete one page | **`FILL`** |
| Collection dates | **`FILL`** |
| Number of sessions | **`FILL`** |
| Pages excluded as illegible | **`FILL`** |

> 500 pages per writer is the figure a reader will not believe on sight.
> It is defensible, but only if "page" is defined. Fill the four rows
> above until the number becomes plausible, and quote total character
> count in preference to page count — it is the better size statistic and
> it is directly comparable to other OCR corpora.

### Writers

One row per writer. Identifiers are pseudonymous.

| writer_id | Age band | Education | District | Dominant hand | Pages |
| --- | --- | --- | --- | --- | --- |
| W01 | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** |
| W02 | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** |
| W03 | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** | **`FILL`** |

### Deliberate variation

Section IV-B states that variation in writing speed, pen type, and
print-versus-cursive style was introduced across sessions. Give counts per
condition, otherwise the claim is not checkable.

| Condition | Levels | Pages per level |
| --- | --- | --- |
| Pen type | **`FILL`** | **`FILL`** |
| Paper type | **`FILL`** | **`FILL`** |
| Writing speed | **`FILL`** | **`FILL`** |
| Print vs cursive | **`FILL`** | **`FILL`** |

### Annotation

- Ground truth: line transcriptions.
- Transcribed by one team member, spot-checked by a second.
- Fraction spot-checked: **`FILL`**
- Full inter-annotator agreement was **not** measured. Stated in the paper
  and repeated here.
- Transcription conventions (punctuation, numerals, unclear characters):
  **`FILL`**

### Splits

| Split | Count | Notes |
| --- | --- | --- |
| Train | 1,050 | |
| Validation | 150 | |
| Test | 300 | self-collected only |

**Not writer-disjoint.** The same three writers appear in train and test,
so the published 12% CER is an estimate for familiar handwriting. With
three writers the natural fix is leave-one-writer-out: train on two, test
on the third, three times. `eval/ocr_cer.py --group-by writer_id` supports
this once `manifest.csv` carries `writer_id`.

### Manifest

`data/handwriting/manifest.csv`, one row per image:

```
image_id,writer_id,session,pen,paper,style,speed,words,chars,split
```

`writer_id` is the field that matters most. Section VII already identifies
it as the key missing metadata; adding it retroactively is what makes
writer-disjoint evaluation possible at all.

### Consent and ethics

- Consent obtained from all three contributors for release of the images:
  **`FILL`** (obtain one signed line each before publishing).
- Images contain no personal data beyond the handwriting itself:
  **`FILL`** — confirm no names, addresses or identifying content appears
  in the written passages.

### Licence

Released under **CC BY 4.0**. Not covered by the repository's MIT licence,
which applies to code only.

---

## 2. Third-party datasets

| Dataset | Ref | Use | URL | Accessed | Licence | Size used |
| --- | --- | --- | --- | --- | --- | --- |
| **BanglaWriting** | **uncited** | **OCR training — the committed adapter** | mendeley.com/datasets/hf6sf8zrkc | **`FILL`** | **`FILL`** | **`FILL`** — read from `data/processed/banglawriting/train.jsonl` |
| Ekush | [28] | OCR training *as described in the paper* | github.com/ShahariarRabby/ekush | **`FILL`** | **`FILL`** | 6,000 images, stratified by grapheme root (`bhasha/data/ekush_sampling.py`) |
| Bengali.AI graphemes | [29] | Labelling reference | kaggle.com/c/bengaliai-cv19 | **`FILL`** | Competition rules — **check redistribution terms** | reference only |
| Kazi Nazrul Islam corpus | [21] | Pre-training | **`FILL`** | **`FILL`** | Public domain — **`FILL`** basis | **`FILL`** tokens |
| Rabindranath Tagore corpus | [22] | Pre-training | **`FILL`** | **`FILL`** | Public domain — **`FILL`** basis | **`FILL`** tokens |
| Spelling checker v1 | [24] | Sanity check | kaggle.com/datasets/mahadivai/spelling-checker-v1 | **`FILL`** | **`FILL`** | — |
| Bangla word frequency 80k | [25] | Sanity check | kaggle.com/datasets/mdraselsarker/bangla-word-frequency-dataset-80k | **`FILL`** | **`FILL`** | — |
| Morphological analysis gold | [26] | Sanity check | kaggle.com/datasets/estiakruddro/... | **`FILL`** | **`FILL`** | — |
| Bengali sentences (OSCAR) | [27] | Sanity check | kaggle.com/datasets/ibraheemmoosa/... | **`FILL`** | **`FILL`** | — |

> Kaggle datasets each carry their own licence and several are
> non-commercial or unspecified. References [21]–[27] currently cite these
> with no URL, access date or licence, which is not a reproducible data
> citation. Every **`FILL`** in the licence column is a redistribution
> question that must be answered before release, not after.

> **BanglaWriting is the dataset behind the committed OCR adapter, and the
> paper does not cite it.** Section IV-C describes Phase 3 as Ekush plus
> self-collected pages. The Phase-3 loader, the OCR evaluation script and
> the production model-path resolver all point at
> `data/processed/banglawriting` and `models/ocr_adapters/banglawriting_adapter`.
> The full evidence table is in `docs/ERRATA.md` §A10. Its URL, access date,
> licence and size must be filled in above before release, and it must be
> added to the reference list. Redistributing or building on a dataset the
> paper never names is the kind of omission that is cheap to fix now and
> expensive later.

**Ekush domain note.** Ekush is isolated handwritten *characters*; the
self-collected material is running text. Roughly 85% of the OCR training
set is therefore isolated characters while 100% of the test set is running
text. This gap plausibly bears on the diacritic-placement errors in
Section V-B, since isolated-character training under-specifies exactly
that. See `docs/ERRATA.md` B12 — and note that the 85% figure describes
the composition the *paper* claims, not the BanglaWriting run that
produced the committed adapter.

**Handwriting manifest.** `data/handwriting/manifest.csv` is the file that
makes writer-disjoint evaluation possible. Generate the template and
validate an existing manifest with:

```bash
python -m bhasha.data.manifest --template data/handwriting/manifest.csv
python -m bhasha.data.manifest --validate data/handwriting/manifest.csv
```

The validator reports whether train and test actually share writers rather
than taking Section VI-D's word for it. Required columns are `image_id`,
`writer_id`, `split` and `source`; the recommended columns record the
session, pen type and print-versus-cursive variation Section IV-B says was
deliberately introduced but that nothing in the repository preserved.

---

## 3. Text corpus processing

| Step | Detail |
| --- | --- |
| Normalisation | NFKC |
| Cleaning | stray Latin characters stripped |
| Final size | ~6.6M tokens |
| Split | 80/10/10 **by document**, so no author appears on both sides |
| Packing | 512-token sequences |
| Split manifests | `data/splits/{train,val,test}.txt` — **`FILL`**, commit these so the by-document claim is checkable rather than asserted |

Balance across genre, register and time period was **not** controlled.
Disclosed in Section VI-D.

### Checksums

Replace the opaque `text dataset.rar` blob:

```bash
unrar x "text dataset.rar" data/text_raw/
git rm --cached "text dataset.rar"
cd data/text_raw && sha256sum * > ../text_raw.sha256
```

Then either commit the plain files, or publish the archive and commit only
the checksum file plus a download script.

---

## 4. Grading dataset

| Field | Value |
| --- | --- |
| Total pairs | 500 |
| Seed | 200 self-generated question-answer pairs from Bangla literature |
| Expansion | human-written feedback |
| Train / validation | 400 / 100 |
| Held out, never used for tuning | 50 |
| Generation procedure | **`FILL`** — which model produced the synthetic pairs, under what prompt? |

The generation procedure matters: if the synthetic pairs came from a model
also being evaluated, that is a confound worth stating.
