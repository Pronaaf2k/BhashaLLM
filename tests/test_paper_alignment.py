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


# ------------------------------------------------------- Table II budget


@pytest.fixture(scope="module")
def budget():
    return _load("memory_budget_mod", "eval/memory_budget.py")


def test_full_finetuning_reaches_the_papers_170gb(budget):
    """Sec. III-C: 'putting the total near 170 GB'."""
    payload = budget.table_ii()
    full = next(r for r in payload["rows"] if r["method"] == "full")
    assert 160 <= full["total_gb"] <= 185
    assert not full["fits_in_16gb"]


def test_errata_b5_multiplier_is_not_four(budget):
    """The paper says the optimiser term is 4x the weight size. It is not.

    fp32 exp_avg + fp32 exp_avg_sq + fp32 master copy = 12 bytes per
    parameter against 2 bytes stored in fp16, i.e. 6x, plus 1x for fp16
    gradients.
    """
    e = budget.table_ii()["errata_b5_check"]
    assert e["stated_multiplier"] == 4
    assert e["computed_multiplier_over_fp16_weights"] == pytest.approx(7.0, abs=0.1)


def test_only_qlora_fits_in_16gb(budget):
    """Table II's whole argument: quantising the base is the deciding step."""
    rows = {r["method"]: r for r in budget.table_ii()["rows"]}
    assert rows["qlora"]["fits_in_16gb"]
    for method in ("full", "lora", "adapters", "prefix"):
        assert not rows[method]["fits_in_16gb"], f"{method} should not fit"


def test_trainable_counts_reproduce_table_ii(budget):
    """Table II: adapters 42M, prefix tuning 3.3M."""
    rows = {r["method"]: r for r in budget.table_ii()["rows"]}
    assert rows["adapters"]["trainable_params"] == pytest.approx(42e6, rel=0.05)
    assert rows["prefix"]["trainable_params"] == pytest.approx(3.3e6, rel=0.05)


def test_quantising_the_base_is_what_saves_the_memory(budget):
    """LoRA and QLoRA train identical parameters; only the base differs."""
    rows = {r["method"]: r for r in budget.table_ii()["rows"]}
    assert (rows["lora"]["trainable_params"]
            == rows["qlora"]["trainable_params"])
    assert (rows["lora"]["terms_gb"]["frozen_weights"]
            > 3 * rows["qlora"]["terms_gb"]["frozen_weights"])


# -------------------------------------------------- hybrid OCR pipeline


def test_script_check_exempts_the_danda():
    """U+0964 is Devanagari-block but is the Bangla sentence terminator."""
    from bhasha.ocr.hybrid_pipeline import script_ok

    assert script_ok("বাংলা ভাষা।")
    assert script_ok("বাংলা ভাষা॥")
    assert not script_ok("बांग्ला भाषा")          # Devanagari letters
    assert not script_ok("plain english only")    # no Bangla at all
    assert not script_ok("")


def test_correction_rejects_a_rewrite_but_accepts_a_repair():
    """Sec. IV-D: a model must not inflate its correction rate by editing
    aggressively. The guardrail is tested with a stub LLM so no weights
    are needed."""
    from bhasha.ocr.hybrid_pipeline import HybridOCRPipeline

    class StubManager:
        def __init__(self, reply):
            self.reply = reply

        def format_chatml(self, messages):
            return messages[0]["content"]

        def generate(self, prompt, **kw):
            return {"text": self.reply}

    reference = "বাংলাদেশের রাজধানী ঢাকা।"
    noisy = "বাংলাদেশের রাজধনী ঢাকা।"

    # A one-character repair is accepted.
    p = HybridOCRPipeline(manager=StubManager(reference))
    text, applied, reason, dist = p.correct_line(noisy)
    assert applied and reason is None and text == reference and dist > 0

    # A wholesale rewrite is rejected and the original kept.
    p = HybridOCRPipeline(manager=StubManager("সম্পূর্ণ ভিন্ন একটি বাক্য যা মূলের সাথে মেলে না একেবারেই"))
    text, applied, reason, _ = p.correct_line(noisy)
    assert not applied
    assert reason == "rewrite_exceeds_max_edit_ratio"
    assert text == noisy

    # A correction that drifts into Devanagari is rejected.
    p = HybridOCRPipeline(manager=StubManager("बांग्लादेश की राजधानी"))
    text, applied, reason, _ = p.correct_line(noisy)
    assert not applied and reason == "script_invalid" and text == noisy

    # An empty correction is rejected.
    p = HybridOCRPipeline(manager=StubManager("   "))
    _, applied, reason, _ = p.correct_line(noisy)
    assert not applied and reason == "empty_correction"


def test_identical_correction_is_not_counted_as_applied():
    """A no-op must not be recorded as a correction; it would inflate the
    numerator of the correction rate."""
    from bhasha.ocr.hybrid_pipeline import HybridOCRPipeline

    class Echo:
        def format_chatml(self, m):
            return m[0]["content"]

        def generate(self, prompt, **kw):
            return {"text": "বাংলাদেশের রাজধানী ঢাকা।"}

    p = HybridOCRPipeline(manager=Echo())
    text, applied, reason, dist = p.correct_line("বাংলাদেশের রাজধানী ঢাকা।")
    assert not applied and dist == 0 and reason is None


def test_pipeline_rejects_an_unknown_detector():
    from bhasha.ocr.hybrid_pipeline import HybridOCRPipeline

    with pytest.raises(ValueError):
        HybridOCRPipeline(detector="magic")


# ------------------------------------------------------- blind rating


@pytest.fixture(scope="module")
def sheets():
    return _load("make_rating_sheets_mod", "eval/make_rating_sheets.py")


def _fake_models(n_models=3, n_items=6):
    return {
        f"model_{chr(ord('a') + m)}": [
            {"item_id": f"i{i}", "output": f"বাংলা লেখা {m}{i}",
             "prompt": f"p{i}", "reference": f"r{i}"}
            for i in range(n_items)
        ]
        for m in range(n_models)
    }


def test_blinding_hides_identity_and_is_reversible(sheets):
    rows, sheet, m = sheets.build(_fake_models(), ["r1", "r2"], 42, "translation")
    codes = set(m["system_code_to_model"])
    assert codes == {"SYS_A", "SYS_B", "SYS_C"}
    # No row carries a real model name.
    assert all(r["system_code"] in codes for r in rows)
    # The map inverts cleanly.
    assert len(set(m["system_code_to_model"].values())) == 3


def test_output_order_is_randomised_per_item_not_once(sheets):
    """A single global shuffle leaves each system in a fixed position, which
    a rater notices within a dozen items."""
    _, _, m = sheets.build(_fake_models(n_items=12), ["r1"], 42, "translation")
    orders = list(m["per_item_system_order"].values())
    assert len(set(tuple(o) for o in orders)) > 1, "order never changed"


def test_blinding_is_reproducible_from_the_seed(sheets):
    _, _, a = sheets.build(_fake_models(), ["r1"], 42, "t")
    _, _, b = sheets.build(_fake_models(), ["r1"], 42, "t")
    _, _, c = sheets.build(_fake_models(), ["r1"], 7, "t")
    assert a["system_code_to_model"] == b["system_code_to_model"]
    assert a["per_item_system_order"] == b["per_item_system_order"]
    assert a["system_code_to_model"] != c["system_code_to_model"]


def test_row_count_is_items_x_systems_x_dimensions_x_raters(sheets):
    rows, _, m = sheets.build(_fake_models(3, 6), ["r1", "r2"], 42, "t")
    assert len(rows) == 6 * 3 * len(sheets.DIMENSIONS) * 2
    assert m["n_items"] == 6


def test_items_missing_a_system_are_dropped(sheets):
    """An item one system did not answer cannot be compared across systems."""
    by_model = _fake_models(3, 4)
    by_model["model_c"] = by_model["model_c"][:2]   # two items missing
    _, _, m = sheets.build(by_model, ["r1"], 42, "t")
    assert m["n_items"] == 2
    assert len(m["items_dropped_incomplete"]) == 2


def test_self_identifying_outputs_are_flagged(sheets):
    """Blinding fails if an output names its own model family."""
    recs = [{"item_id": "i0", "output": "As an AI language model, I cannot."},
            {"item_id": "i1", "output": "বাংলা লেখা"}]
    hits = sheets.check_fingerprints("model_a", recs)
    assert len(hits) == 1 and hits[0]["item_id"] == "i0"


def test_length_separability_warns_when_systems_are_sortable(sheets):
    by_model = {"short": [{"item_id": "i0", "output": "ক" * 10}],
                "long": [{"item_id": "i0", "output": "ক" * 500}]}
    assert sheets.length_separability(by_model)["warning"] is not None


# ---------------------------------------------------- tokenizer sanity


def test_tokenizer_sanity_script_helpers():
    mod = _load("tokenizer_sanity_mod", "eval/tokenizer_sanity.py")
    assert mod.is_bengali("ক") and not mod.is_bengali("क")
    assert mod.is_devanagari("क")
    # The danda is exempt for the same reason as everywhere else.
    assert not mod.is_devanagari("।")
    assert mod._script_of("বাংলা") == "bengali"
    assert mod._script_of("बांग्ला") == "devanagari"
    assert mod._script_of("বাংলা बांग्ला") == "mixed"


# -------------------------------------------------------- web frontend


def test_ui_is_opt_in_not_mounted_by_default(monkeypatch=None):
    """Sec. III-F: the frontend 'is not loaded unless explicitly opened'.

    Mounting it by default would contradict the sentence the file exists to
    implement.
    """
    import os
    from bhasha.app import routes_ui

    saved = os.environ.pop(routes_ui.ENV_FLAG, None)
    try:
        assert routes_ui.ui_enabled() is False
        for value in ("1", "true", "YES", "on"):
            os.environ[routes_ui.ENV_FLAG] = value
            assert routes_ui.ui_enabled() is True, value
        os.environ[routes_ui.ENV_FLAG] = "0"
        assert routes_ui.ui_enabled() is False
    finally:
        os.environ.pop(routes_ui.ENV_FLAG, None)
        if saved is not None:
            os.environ[routes_ui.ENV_FLAG] = saved


def test_frontend_file_exists_and_calls_only_the_v1_api():
    """Sec. III-F: it 'talks to the same backend'.

    A second inference path in the page would be a second thing to keep in
    sync with test_models.py, and the first place the two would diverge.
    """
    html = (ROOT / "bhasha/app/static/index.html").read_text(encoding="utf-8")
    for endpoint in ("/api/v1/generate", "/api/v1/grade",
                     "/api/v1/ocr", "/api/v1/status"):
        assert endpoint in html, f"{endpoint} not called by the frontend"

    # No build step, no framework, no third-party fetch. "Does not compete
    # with the language models for GPU memory" only holds for a static file.
    lowered = html.lower()
    for forbidden in ("cdn.", "unpkg", "jsdelivr", "googleapis",
                      "import react", "require("):
        assert forbidden not in lowered, f"frontend pulls in {forbidden}"


def test_frontend_script_check_exempts_the_danda():
    """The page flags Devanagari drift; it must not flag the danda.

    Same rule as eval/script_integrity.py and hybrid_pipeline.script_ok.
    Three implementations of one definition is already one too many, so
    the third is tested against the same cases.
    """
    html = (ROOT / "bhasha/app/static/index.html").read_text(encoding="utf-8")
    assert "0x0964" in html and "0x0965" in html, \
        "frontend script check does not exempt the danda"


# ------------------------------------------------------- corpus audit


@pytest.fixture(scope="module")
def audit():
    return _load("audit_text_corpus_mod", "scripts/audit_text_corpus.py")


def test_audit_reads_entry_names_without_a_rar_tool(audit):
    """RAR filenames are plain bytes in the header, so structure is
    recoverable on a machine with no decoder installed."""
    archive = ROOT / "text dataset.rar"
    if not archive.exists():
        pytest.skip("text dataset.rar not present")
    names = audit.entry_names_from_bytes(archive.read_bytes())
    assert len(names) > 50
    assert any(n.endswith(".txt") for n in names)


def test_audit_flags_missing_author_metadata(audit, tmp_path):
    """Sec. IV-C claims an author-disjoint split. Numbered files cannot
    support it, and the audit has to say so."""
    root = tmp_path / "text dataset"
    root.mkdir()
    for i in range(1, 6):
        (root / f"{i}.txt").write_text("বাংলা লেখা।" * 20, encoding="utf-8")

    analysis = audit.analyse_extracted(tmp_path, None)
    assert analysis["filenames_are_numeric_only"] is True
    assert analysis["top_level_directories"] == ["text dataset"]

    findings = " ".join(audit.compare_to_paper(analysis)["findings"])
    assert "author" in findings.lower()
    assert "same author appears on both sides" in findings


def test_audit_flags_a_corpus_far_smaller_than_table_iv(audit, tmp_path):
    root = tmp_path / "text dataset"
    root.mkdir()
    (root / "1.txt").write_text("বাংলা।" * 100, encoding="utf-8")

    cmp_ = audit.compare_to_paper(audit.analyse_extracted(tmp_path, None))
    assert cmp_["paper_claim"] == 6_600_000
    assert cmp_["ratio_to_claim"] < 0.5
    assert any("not reproducible" in f for f in cmp_["findings"])


def test_audit_reports_no_findings_for_a_conforming_corpus(audit, tmp_path):
    """The audit must not fire on a corpus that does satisfy the paper."""
    for author in ("nazrul", "tagore"):
        d = tmp_path / author
        d.mkdir()
        # ~3 chars/token, so ~3.3M tokens per author reaches Table IV's 6.6M.
        (d / "collected.txt").write_text("ক" * 10_000_000, encoding="utf-8")

    cmp_ = audit.compare_to_paper(audit.analyse_extracted(tmp_path, None))
    assert cmp_["findings"] == [], cmp_["findings"]


# ------------------------------------------------- ekush label mapping


def test_ekush_mapping_has_the_documented_122_classes():
    from bhasha.eval.ekush_mapping import CLASS_GROUPS, DEFAULT_MAP

    # Ekush (ref [28]): 10 numerals + 11 vowels + 39 consonants
    # + 10 modifiers + 52 compounds.
    assert len(DEFAULT_MAP) == 122
    sizes = {k: hi - lo for k, (lo, hi) in CLASS_GROUPS.items()}
    assert sizes == {"numeral": 10, "vowel": 11, "consonant": 39,
                     "modifier": 10, "compound": 52}


def test_get_label_text_accepts_every_form_the_call_sites_use():
    from bhasha.eval.ekush_mapping import get_label_text

    assert get_label_text(0) == "০"
    assert get_label_text(21) == "ক"
    assert get_label_text("21") == "ক"
    assert get_label_text("/data/ekush/21/img_001.png") == "ক"
    # Already Bangla: returned untouched, so it is safe to apply to the
    # JSONL `text` field the two eval modules actually read.
    assert get_label_text("ক্ষ") == "ক্ষ"
    assert get_label_text("") == ""
    # Unmappable input must not raise mid-evaluation.
    assert get_label_text("not_a_class") == "not_a_class"


def test_the_two_previously_broken_modules_can_resolve_their_import():
    """bhasha/eval/{all_ocr,ocr_models}.py did `from ekush_mapping import ...`
    with no such module anywhere. Both raised ModuleNotFoundError."""
    assert (ROOT / "bhasha/eval/ekush_mapping.py").exists()
    for name in ("all_ocr.py", "ocr_models.py"):
        src = (ROOT / "bhasha/eval" / name).read_text(encoding="utf-8")
        assert "from ekush_mapping import" in src, f"{name} changed its import"


# ------------------------------------------- llm outputs -> JSONL


@pytest.fixture(scope="module")
def convert():
    return _load("convert_llm_outputs_mod", "scripts/convert_llm_outputs.py")


SAMPLE_MD = """# Test Model

## Benchmark Outputs (Ollama Backend)

### Translation

**Prompt:**
> Translate this into Bangla:
>
> 'Hello world.'

**Response:** (2.86s)

বিশ্ব, তোমাকে স্বাগতম।

---
### OCR Fix

**Prompt:**
> Fix the errors:
>
> 'আিম বংলাদশ এ থািক।' > (Expected: আমি বাংলাদেশে থাকি।)

**Response:** (0.42s)

আমি বাংলাদেশে থাকি।

---
"""


def test_converter_extracts_prompt_response_and_latency(convert):
    parsed = convert.parse_markdown(SAMPLE_MD)
    assert [p["category"] for p in parsed] == ["Translation", "OCR Fix"]
    assert parsed[0]["seconds"] == 2.86
    assert "স্বাগতম" in parsed[0]["response"]
    assert "**Response:**" not in parsed[0]["response"]


def test_converter_recovers_the_leaked_reference(convert):
    """The OCR-Fix prompt embeds `(Expected: ...)`, which is both the only
    available reference and the reason those scores are meaningless."""
    records = convert.build_records("Test", convert.parse_markdown(SAMPLE_MD))
    ocr = next(r for r in records if r["task"] == "ocr_correction")
    assert ocr["reference"] == "আমি বাংলাদেশে থাকি।"
    assert ocr["noisy"] == "আিম বংলাদশ এ থািক।"
    # The leak, which docs/ERRATA.md C17 is about: the reference the model
    # is scored against was visible in its own prompt.
    assert ocr["reference"] in ocr["prompt"]


def test_records_carry_both_keys_the_eval_scripts_read(convert):
    """script_integrity.py reads `output`; text_metrics.py and
    ocr_correction.py read `hypothesis`. One file must feed all three."""
    records = convert.build_records("Test", convert.parse_markdown(SAMPLE_MD))
    for r in records:
        assert r["output"] == r["hypothesis"]


def test_every_record_is_stamped_with_the_real_provenance(convert):
    records = convert.build_records("Test", convert.parse_markdown(SAMPLE_MD))
    p = records[0]["provenance"]
    assert p["backend"] == "ollama"
    assert "Q4_K_M" in p["quantisation"]
    # Sec. IV-C says 0.7; the runner used 0.3 (ERRATA C13).
    assert p["temperature"] == 0.3
    # Four prompts, one per category (ERRATA C12).
    assert p["n_items_per_task"] == 1


def test_llama32_tag_mismatch_is_flagged(convert):
    """The finding the whole of ERRATA A00 rests on."""
    info = convert.OLLAMA_TAGS["Llama_3.2_11B"]
    assert info["tag"] == "llama3.2:latest"
    assert info["mismatch"] is True
    assert "3b" in info["resolves_to"].lower()
    # Every tag that matches its label must not be flagged.
    for name in ("Llama_3.1_8B", "Mistral_7B", "Nemo_12B",
                 "Gemma_9B", "Qwen_1.5B"):
        assert convert.OLLAMA_TAGS[name]["mismatch"] is False, name


def test_registry_records_the_benchmark_execution():
    reg = json.loads((ROOT / "benchmarks/model_registry.json").read_text(
        encoding="utf-8"))
    ex = reg["benchmark_execution"]
    assert ex["default_quantisation"] == "Q4_K_M"
    assert ex["decoding_as_run"]["temperature"] == 0.3
    assert ex["decoding_as_published"]["temperature"] == 0.7
    assert ex["n_items_per_task"] == 1

    llama = next(m for m in reg["models"] if "Llama-3.2-11B" in m["model_id"])
    assert llama["ollama_tag"] == "llama3.2:latest"
    assert "A00" in llama["notes"]

    # Llama-3-8B has Table V and VI scores but was never run.
    l3 = next(m for m in reg["models"] if "Meta-Llama-3-8B" in m["model_id"])
    assert l3["quantisation"] == "NOT RUN"


def test_model_registry_covers_the_nine_benchmarked_models():
    reg = json.loads((ROOT / "benchmarks/model_registry.json").read_text(
        encoding="utf-8"))
    ids = [m["model_id"] for m in reg["models"]]
    # Table I lists nine architectures; the registry adds XGLM (Sec. III-B
    # selection comparison) and the OCR base, hence 11 entries.
    assert len(ids) >= 11
    assert reg["decoding"]["latency_measurements"]["max_new_tokens"] == 100
    assert reg["decoding"]["text"]["max_new_tokens"] == 256
    llama = next(m for m in reg["models"] if "Llama-3.2-11B" in m["model_id"])
    # ERRATA B1: this model was never fine-tuned. The registry must say so.
    assert "inference" in llama["notes"].lower()
