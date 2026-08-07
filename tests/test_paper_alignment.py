"""Tests for the components added to align the repository with the paper.

Everything here is a pure function or a config load, so the suite runs on
CPU in under a second and needs no model weights, no GPU and no network.
That is deliberate: a test suite that requires a 16 GB card is a test suite
nobody runs.

Covers:
  * Table III fidelity of every committed phase config
  * the by-document corpus split and the Sec. IV-B normalisation
  * grapheme-root stratification actually retaining rare roots
  * the OCR-correction rate denominators (docs/ERRATA.md B11)
  * the confidence definition (docs/ERRATA.md B11)
  * the energy arithmetic that docs/ERRATA.md B4 found wrong
  * manifest validation and the writer-disjointness audit
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, relpath):
    """Import a script from eval/ by path (they are scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- Table III


def test_table_iii_defaults_without_config():
    from bhasha.config import PhaseConfig

    cfg = PhaseConfig.load(None)
    lora = cfg.lora_kwargs()
    assert lora["r"] == 16
    assert lora["lora_alpha"] == 32
    assert lora["lora_dropout"] == 0.05
    # The value that produces Table II's 9.4M trainable parameters.
    assert lora["target_modules"] == ["q_proj", "v_proj"]
    assert cfg.training["learning_rate"] == pytest.approx(2e-4)
    assert cfg.training["per_device_train_batch_size"] == 1
    assert cfg.training["gradient_accumulation_steps"] == 4
    assert cfg.training["warmup_steps"] == 50
    assert cfg.training["max_steps"] == 500
    assert cfg.max_seq_length == 512
    assert cfg.training["optim"] == "adamw_torch"
    assert cfg.seed == 42
    assert cfg.effective_batch == 4  # Table III / Sec. IV-C


@pytest.mark.parametrize("path,phase", [
    ("configs/phase1_bangla_pt.yaml", "1_bangla_pt"),
    ("configs/phase2_grading_sft.yaml", "2_grading_sft"),
    ("configs/phase3_ocr_sft.yaml", "3_ocr_sft"),
])
def test_committed_configs_load_and_match_table_iii(path, phase):
    pytest.importorskip("yaml")
    from bhasha.config import PhaseConfig

    cfg = PhaseConfig.load(ROOT / path)
    assert cfg.phase == phase
    assert cfg.seed == 42
    assert cfg.quantization["bnb_4bit_quant_type"] == "nf4"
    assert cfg.lora_kwargs()["target_modules"] == ["q_proj", "v_proj"]
    assert cfg.effective_batch == 4
    assert cfg.summary_path.name == f"phase{phase[0]}_summary.json"


def test_phase3_carries_the_two_documented_deviations():
    """Paper Sec. IV-C: 2000 steps not 500, lr 1e-4 not 2e-4."""
    pytest.importorskip("yaml")
    from bhasha.config import PhaseConfig

    cfg = PhaseConfig.load(ROOT / "configs/phase3_ocr_sft.yaml")
    assert cfg.training["max_steps"] == 2000
    assert cfg.training["learning_rate"] == pytest.approx(1e-4)


def test_legacy_target_modules_are_recoverable():
    """Backward compatibility: the pre-Table-III set must stay reachable."""
    from bhasha.config import PhaseConfig, LEGACY_TARGET_MODULES

    cfg = PhaseConfig.load(None, lora={"target_modules": "legacy"})
    assert cfg.lora_kwargs()["target_modules"] == LEGACY_TARGET_MODULES
    assert len(LEGACY_TARGET_MODULES) == 7


def test_max_steps_and_epochs_are_never_both_set():
    """HF silently ignores epochs when max_steps > 0; an ambiguous log is worse
    than a missing one."""
    from bhasha.config import PhaseConfig

    kw = PhaseConfig.load(None).training_kwargs("/tmp/out")
    assert "max_steps" in kw
    assert "num_train_epochs" not in kw


# ----------------------------------------------------------- run summaries


def test_epochs_covered_matches_the_errata_b2_arithmetic():
    """5.28M tokens / 512 = 10,312 sequences; 500 steps x 4 = 0.19 epochs.

    docs/ERRATA.md B2: the paper reports 0.8 for the same run.
    """
    from bhasha.utils.run_summary import epochs_covered

    assert epochs_covered(500, 4, 10312) == pytest.approx(0.1939, abs=1e-3)
    # And the figure the paper's own 2,578-sequence claim would give:
    assert epochs_covered(500, 4, 2578) == pytest.approx(0.7758, abs=1e-3)
    assert epochs_covered(500, 4, None) is None


def test_run_summary_survives_without_cuda():
    from bhasha.config import PhaseConfig
    from bhasha.utils.run_summary import RunSummary

    cfg = PhaseConfig.load(None)
    s = RunSummary(cfg, n_train_sequences=100).start().finish()
    assert s.payload["seed"] == 42
    assert s.payload["effective_batch"] == 4
    assert "peak_vram_gb" in s.payload  # None on CPU, but present
    assert isinstance(s.report(), str)


# ------------------------------------------------------------ text corpus


def test_normalisation_strips_latin_but_keeps_the_danda():
    """The danda is the Bangla sentence terminator, not stray Latin."""
    from bhasha.data.text_corpus import normalise

    out = normalise("বাংলা hello ভাষা। test")
    assert "hello" not in out and "test" not in out
    assert "।" in out
    assert "বাংলা" in out and "ভাষা" in out


def test_normalisation_keeps_digits():
    from bhasha.data.text_corpus import strip_latin
    assert "1971" in strip_latin("সাল 1971 abc")


def test_split_is_by_document_and_deterministic():
    from bhasha.data.text_corpus import split_by_document

    docs = [{"doc_id": f"d{i}", "author": f"a{i}", "text": "x"} for i in range(20)]
    a = split_by_document(docs, seed=42, group_key=None)
    b = split_by_document(docs, seed=42, group_key=None)
    assert [d["doc_id"] for d in a["train"]] == [d["doc_id"] for d in b["train"]]
    assert len(a["train"]) == 16 and len(a["val"]) == 2 and len(a["test"]) == 2

    ids = lambda k: {d["doc_id"] for d in a[k]}
    assert not (ids("train") & ids("test"))
    assert not (ids("train") & ids("val"))


def test_two_author_corpus_reports_the_grouping_fallback():
    """Nazrul + Tagore is two authors; an author-disjoint 3-way split is
    impossible and the code must say so rather than pretend otherwise."""
    from bhasha.data.text_corpus import split_by_document

    docs = ([{"doc_id": f"n{i}", "author": "nazrul", "text": "x"} for i in range(5)]
            + [{"doc_id": f"t{i}", "author": "tagore", "text": "x"} for i in range(5)])
    out = split_by_document(docs, seed=42, group_key="author")
    assert out["_meta"][0]["grouping_fallback"] is not None
    assert "author" in out["_meta"][0]["grouping_fallback"]


# --------------------------------------------------------- Ekush sampling


def test_grapheme_root_strips_matra_and_hasant():
    from bhasha.data.ekush_sampling import grapheme_root

    assert grapheme_root("কা") == "ক"      # vowel sign removed
    assert grapheme_root("ক্ক") == "ক"     # conjunct -> leading root
    assert grapheme_root("কি") == "ক"
    assert grapheme_root("") == "<empty>"


def test_rare_roots_survive_stratified_sampling():
    """Paper Sec. IV-C's stated purpose: 'so that rare conjuncts survived
    sampling'. Proportional allocation alone would drop them."""
    from bhasha.data.ekush_sampling import stratified_sample

    records = ([{"text": "ক", "i": i} for i in range(5000)]
               + [{"text": "ঞ", "i": i} for i in range(12)])
    out = stratified_sample(records, n_total=600, min_per_class=8, seed=42)
    assert out["report"]["strata_lost"] == 0
    drawn = [r for r in out["records"] if r["text"] == "ঞ"]
    assert len(drawn) >= 8, "the rare root did not survive sampling"
    assert out["report"]["drawn"] == 600


def test_allocation_never_exceeds_availability():
    from bhasha.data.ekush_sampling import allocate

    counts = {"a": 3, "b": 100, "c": 1}
    alloc = allocate(counts, 50, min_per_class=8)
    for k, v in alloc.items():
        assert v <= counts[k], f"allocated {v} of {counts[k]} available for {k}"
    assert sum(alloc.values()) <= 50


# --------------------------------------------------------- OCR correction


@pytest.fixture(scope="module")
def correction():
    return _load("ocr_correction_mod", "eval/ocr_correction.py")


def test_perfect_correction(correction):
    item = correction.score_item({
        "reference": "বাংলা", "noisy": "বাংলর", "hypothesis": "বাংলা"})
    agg = correction.aggregate([item])["rates"]
    assert agg["correction_rate"] == 1.0
    assert agg["over_correction_rate"] == 0.0
    assert agg["net_error_reduction"] == 1.0


def test_no_op_corrector_scores_zero_not_undefined(correction):
    item = correction.score_item({
        "reference": "বাংলা", "noisy": "বাংলর", "hypothesis": "বাংলর"})
    r = correction.aggregate([item])["rates"]
    assert r["correction_rate"] == 0.0
    assert r["over_correction_rate"] == 0.0
    # No edits made: the false-positive denominator is zero, so the rate is
    # undefined rather than 0.0. Reporting 0.0 would flatter a do-nothing model.
    assert r["false_positive_rate"] is None


def test_aggressive_corrector_is_penalised(correction):
    """A model that rewrites correct text must show over-correction.

    Paper Sec. IV-D: the rates are tracked separately 'so that a model
    cannot inflate its apparent correction rate simply by editing text
    aggressively'.
    """
    item = correction.score_item({
        "reference": "বাংলা ভাষা",
        "noisy": "বাংলর ভাষা",
        "hypothesis": "বাংলা ভাসা",   # fixed one, broke another
    })
    r = correction.aggregate([item])["rates"]
    assert r["correction_rate"] == 1.0
    assert r["over_correction_rate"] > 0.0
    assert r["false_positive_rate"] > 0.0
    assert r["net_error_reduction"] == pytest.approx(0.0)


def test_every_rate_reports_its_denominator(correction):
    """docs/ERRATA.md B11 exists because Sec. V-B omitted these."""
    item = correction.score_item({
        "reference": "বাংলা", "noisy": "বাংলর", "hypothesis": "বাংলা"})
    rates = correction.aggregate([item])["rates"]
    for name in ("correction_rate", "over_correction_rate",
                 "false_positive_rate"):
        assert f"{name}_denominator" in rates


# ------------------------------------------------------------- confidence


@pytest.fixture(scope="module")
def confidence():
    return _load("confidence_mod", "eval/confidence.py")


def test_confidence_is_geometric_mean_probability(confidence):
    lps = [math.log(0.5), math.log(0.5)]
    assert confidence.sequence_confidence(lps) == pytest.approx(0.5)
    assert confidence.sequence_confidence([0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_is_length_normalised(confidence):
    """Two tokens at p=0.9 and ten tokens at p=0.9 must score the same.

    Without normalisation the raw sequence probability would report the
    longer transcription as less confident purely for being longer, which
    is exactly the comparison Table VII makes.
    """
    short = confidence.sequence_confidence([math.log(0.9)] * 2)
    long = confidence.sequence_confidence([math.log(0.9)] * 10)
    assert short == pytest.approx(long)


def test_empty_generation_is_null_not_zero(confidence):
    assert confidence.sequence_confidence([]) is None
    agg = confidence.aggregate([confidence.summarise_item({"token_logprobs": []})])
    assert agg["n_empty_generations"] == 1


def test_calibration_error_detects_overconfidence(confidence):
    """A model reporting 0.95 while being 50% right must show a large ECE."""
    pairs = [(0.95, 0.5)] * 20
    cal = confidence.calibration(pairs, n_bins=10)
    assert cal["expected_calibration_error"] == pytest.approx(0.45, abs=0.01)


# ----------------------------------------------------------------- energy


@pytest.fixture(scope="module")
def energy():
    return _load("energy_mod", "eval/energy.py")


def test_energy_reproduces_the_papers_600ms_arithmetic(energy):
    """Sec. VI-F: 250 W x 600 ms x 1000 = ~42 Wh. The arithmetic is right."""
    out = energy.compute(600.0, 250.0, 0.476, 1000)
    assert out["energy_wh"] == pytest.approx(41.67, abs=0.1)
    assert out["co2_g"] == pytest.approx(19.8, abs=0.5)


def test_energy_at_the_latency_the_paper_actually_reports(energy):
    """docs/ERRATA.md B4: Sec. V-C reports 1450 ms, giving ~101 Wh, not 42."""
    out = energy.compute(1450.0, 250.0, 0.476, 1000)
    assert out["energy_wh"] == pytest.approx(100.7, abs=0.5)
    ratio = out["energy_wh"] / energy.compute(600.0, 250.0, 0.476, 1000)["energy_wh"]
    assert ratio == pytest.approx(2.42, abs=0.02)


# --------------------------------------------------------------- manifest


def test_manifest_requires_writer_id(tmp_path):
    from bhasha.data.manifest import validate

    rows = [{"image_id": "a", "writer_id": "", "split": "train",
             "source": "self_collected"}]
    result = validate(rows)
    assert not result["ok"]
    assert any("writer_id" in p for p in result["problems"])


def test_manifest_detects_duplicate_image_ids():
    from bhasha.data.manifest import validate

    rows = [{"image_id": "a", "writer_id": "w1", "split": "train", "source": "s"},
            {"image_id": "a", "writer_id": "w2", "split": "test", "source": "s"}]
    assert any("duplicate" in p for p in validate(rows)["problems"])


def test_writer_disjointness_audit_matches_sec_vi_d():
    """Sec. VI-D concedes train and test share writers. Computed, not recalled."""
    from bhasha.data.manifest import audit_writer_disjointness

    overlapping = [
        {"image_id": "1", "writer_id": "A", "split": "train", "source": "s"},
        {"image_id": "2", "writer_id": "A", "split": "test", "source": "s"},
    ]
    out = audit_writer_disjointness(overlapping)
    assert out["writer_disjoint"] is False
    assert out["train_test_overlap"] == ["A"]
    assert "optimistic" in out["interpretation"]

    disjoint = [
        {"image_id": "1", "writer_id": "A", "split": "train", "source": "s"},
        {"image_id": "2", "writer_id": "B", "split": "test", "source": "s"},
    ]
    assert audit_writer_disjointness(disjoint)["writer_disjoint"] is True


def test_manifest_template_round_trips(tmp_path):
    from bhasha.data.manifest import (
        REQUIRED_COLUMNS, read_manifest, validate, write_template)

    p = write_template(tmp_path / "manifest.csv")
    rows = read_manifest(p)
    assert all(c in rows[0] for c in REQUIRED_COLUMNS)
    assert validate(rows)["ok"]


# ------------------------------------------------------- dataset collation


def test_collate_pads_labels_with_ignore_index_not_pad_token():
    """Padding labels with the pad token trains the model to emit padding
    and deflates the reported loss. It is a silent failure, so it is tested."""
    import torch
    from bhasha.data.dataset import IGNORE_INDEX, collate_fn

    batch = [
        {"input_ids": torch.tensor([1, 2, 3]),
         "attention_mask": torch.tensor([1, 1, 1]),
         "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]),
         "attention_mask": torch.tensor([1, 1]),
         "labels": torch.tensor([4, 5])},
    ]
    out = collate_fn(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"][1, 2].item() == IGNORE_INDEX
    assert out["attention_mask"][1, 2].item() == 0


def test_collate_carries_provenance_keys():
    """writer_id must survive collation or the Sec. VI-D breakdown is lost."""
    import torch
    from bhasha.data.dataset import collate_fn

    batch = [{"input_ids": torch.tensor([1]),
              "attention_mask": torch.tensor([1]),
              "labels": torch.tensor([1]),
              "writer_id": "A", "image_id": "img1"}]
    out = collate_fn(batch)
    assert out["writer_id"] == ["A"]
    assert out["image_id"] == ["img1"]


# ------------------------------------------------------- registry sanity


def test_model_registry_covers_the_nine_benchmarked_models():
    reg = json.loads((ROOT / "benchmarks/model_registry.json").read_text(
        encoding="utf-8"))
    ids = [m["model_id"] for m in reg["models"]]
    # Table I lists nine architectures; the registry adds XGLM (Sec. III-B
    # selection comparison) and the OCR base, hence 11 entries.
    assert len(ids) == 11
    assert reg["decoding"]["latency_measurements"]["max_new_tokens"] == 100
    assert reg["decoding"]["text"]["max_new_tokens"] == 256
    llama = next(m for m in reg["models"] if "Llama-3.2-11B" in m["model_id"])
    # ERRATA B1: this model was never fine-tuned. The registry must say so.
    assert "inference" in llama["notes"].lower()
