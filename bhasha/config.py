"""Phase configuration loader — Table III of the paper, in one place.

``configs/phase{1,2,3}*.yaml`` were committed with the paper's Table III
hyperparameters and detailed reconciliation notes, and ``README.md``
documents running training with ``--config configs/phase1_bangla_pt.yaml``.
No code read those files. The trainers hardcoded a different configuration,
so the committed configs were documentation with nothing behind them and
the README command did not work.

This module makes the YAML authoritative. ``PhaseConfig.load`` reads a
config file, applies Table III as the default for anything the file omits,
and exposes ready-built ``BitsAndBytesConfig``/``LoraConfig``/
``TrainingArguments`` keyword dictionaries so all three phases construct
their objects from the same source.

Table III (paper Sec. III-F)
----------------------------
======================  =========================
Quantisation            4-bit (NF4)
LoRA rank (r)           16
LoRA alpha              32
LoRA dropout            0.05
Target modules          q_proj, v_proj
Learning rate           2e-4
Batch size (per GPU)    1
Gradient accumulation   4
Warm-up steps           50
Total training steps    500
Max sequence length     512
Optimizer               AdamW
======================  =========================

Two deviations are documented in the paper itself (Sec. IV-C) and live in
``configs/phase3_ocr_sft.yaml``: 2,000 steps rather than 500, and a
learning rate of 1e-4 rather than 2e-4.

On target modules
-----------------
Table III specifies ``q_proj, v_proj`` only, which is what produces the
9.4M trainable parameters quoted in Table II. The previously committed
trainers targeted all seven projection matrices
(``q,k,v,o,gate,up,down``), which is roughly 4-5x more trainable
parameters and therefore does not reproduce Table II. The default here is
Table III. Pass ``--target-modules`` on the command line or set
``lora.target_modules`` in the YAML to restore the wider set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------- Table III

TABLE_III: Dict[str, Any] = {
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "bias": "none",
    },
    "training": {
        "learning_rate": 2.0e-4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
        "warmup_steps": 50,
        "lr_scheduler_type": "cosine",
        "max_seq_length": 512,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "logging_steps": 10,
    },
    "seed": 42,
}

# The wider target set the repository used before Table III was wired in.
# Kept as a named constant so `--target-modules legacy` reproduces the old
# behaviour exactly rather than requiring the user to retype seven strings.
LEGACY_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _torch_dtype(name: str):
    import torch
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }[str(name).lower()]


@dataclass
class PhaseConfig:
    """A resolved training-phase configuration.

    ``raw`` keeps the merged dictionary so the exact configuration can be
    serialised into the phase summary. A hyperparameter that is not in the
    log is a hyperparameter nobody can check.
    """

    phase: str = "unnamed"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    init_adapter: Optional[str] = None
    seed: int = 42
    quantization: Dict[str, Any] = field(default_factory=dict)
    lora: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Optional[str | Path] = None, **overrides: Any) -> "PhaseConfig":
        """Load a YAML phase config, defaulting anything absent to Table III.

        ``path=None`` returns a pure Table III configuration, which is what
        the trainers use when invoked without ``--config``.
        """
        merged = _deep_merge(TABLE_III, {})
        source = None
        if path is not None:
            import yaml
            source = str(path)
            text = Path(path).read_text(encoding="utf-8")
            merged = _deep_merge(merged, yaml.safe_load(text) or {})
        merged = _deep_merge(merged, overrides or {})

        return cls(
            phase=merged.get("phase", "unnamed"),
            base_model=merged.get("base_model", TABLE_III and "Qwen/Qwen2.5-1.5B-Instruct"),
            init_adapter=merged.get("init_adapter"),
            seed=int(merged.get("seed", 42)),
            quantization=merged.get("quantization", {}),
            lora=merged.get("lora", {}),
            training=merged.get("training", {}),
            data=merged.get("data", {}),
            evaluation=merged.get("evaluation", {}),
            logging=merged.get("logging", {}),
            source_path=source,
            raw=merged,
        )

    # -------------------------------------------------------- derived views

    @property
    def effective_batch(self) -> int:
        """per-device batch x gradient accumulation. Table III gives 4."""
        return int(self.training.get("per_device_train_batch_size", 1)) * int(
            self.training.get("gradient_accumulation_steps", 4)
        )

    @property
    def max_seq_length(self) -> int:
        return int(self.training.get("max_seq_length", 512))

    @property
    def summary_path(self) -> Path:
        """Where the run summary goes. README documents logs/phaseN_summary.json."""
        p = self.logging.get("summary_path") or f"logs/{self.phase}_summary.json"
        p = Path(p)
        return p if p.is_absolute() else BASE_DIR / p

    def bnb_kwargs(self) -> Dict[str, Any]:
        q = self.quantization
        return {
            "load_in_4bit": bool(q.get("load_in_4bit", True)),
            "bnb_4bit_quant_type": q.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_compute_dtype": _torch_dtype(
                q.get("bnb_4bit_compute_dtype", "float16")
            ),
            "bnb_4bit_use_double_quant": bool(
                q.get("bnb_4bit_use_double_quant", True)
            ),
        }

    def lora_kwargs(self) -> Dict[str, Any]:
        l = self.lora
        targets = l.get("target_modules", ["q_proj", "v_proj"])
        if isinstance(targets, str):
            targets = (
                LEGACY_TARGET_MODULES if targets.lower() == "legacy"
                else [t.strip() for t in targets.split(",") if t.strip()]
            )
        return {
            "r": int(l.get("r", 16)),
            "lora_alpha": int(l.get("alpha", l.get("lora_alpha", 32))),
            "lora_dropout": float(l.get("dropout", l.get("lora_dropout", 0.05))),
            "bias": l.get("bias", "none"),
            "task_type": l.get("task_type", "CAUSAL_LM"),
            "target_modules": list(targets),
        }

    def training_kwargs(self, output_dir: str | Path) -> Dict[str, Any]:
        """Keyword arguments for ``transformers.TrainingArguments``.

        Only Table III quantities plus the logging fields are set here.
        Anything phase-specific (evaluation strategy, save cadence) is left
        to the caller, which keeps the mapping from Table III to code
        one-to-one and readable.
        """
        t = self.training
        kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(
                t.get("per_device_train_batch_size", 1)
            ),
            "gradient_accumulation_steps": int(
                t.get("gradient_accumulation_steps", 4)
            ),
            "learning_rate": float(t.get("learning_rate", 2e-4)),
            "warmup_steps": int(t.get("warmup_steps", 50)),
            "lr_scheduler_type": t.get("lr_scheduler_type", "cosine"),
            "gradient_checkpointing": bool(t.get("gradient_checkpointing", True)),
            "optim": t.get("optim", "adamw_torch"),
            "logging_steps": int(t.get("logging_steps", 10)),
            "seed": self.seed,
            "data_seed": self.seed,
            "report_to": "none",
        }
        # max_steps and num_train_epochs are mutually exclusive in practice:
        # HF ignores epochs when max_steps > 0. Set whichever the config
        # specifies and never both, so the log is unambiguous.
        if int(t.get("max_steps", 0) or 0) > 0:
            kwargs["max_steps"] = int(t["max_steps"])
        elif t.get("num_train_epochs"):
            kwargs["num_train_epochs"] = float(t["num_train_epochs"])
        if t.get("weight_decay") is not None:
            kwargs["weight_decay"] = float(t["weight_decay"])
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d

    def describe(self) -> str:
        lk = self.lora_kwargs()
        t = self.training
        return (
            f"phase={self.phase}  base={self.base_model}  seed={self.seed}\n"
            f"  quantisation : 4bit={self.quantization.get('load_in_4bit')} "
            f"type={self.quantization.get('bnb_4bit_quant_type')}\n"
            f"  lora         : r={lk['r']} alpha={lk['lora_alpha']} "
            f"dropout={lk['lora_dropout']} targets={lk['target_modules']}\n"
            f"  training     : lr={t.get('learning_rate')} "
            f"batch={t.get('per_device_train_batch_size')}x"
            f"{t.get('gradient_accumulation_steps')}"
            f"(eff {self.effective_batch}) steps={t.get('max_steps')} "
            f"warmup={t.get('warmup_steps')} sched={t.get('lr_scheduler_type')} "
            f"seqlen={self.max_seq_length} optim={t.get('optim')}\n"
            f"  source       : {self.source_path or 'Table III defaults (no --config)'}"
        )


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch.

    Paper Sec. IV-C: "All runs used seed 42 and were executed once, so the
    losses carry no variance estimate." The first half of that sentence was
    not true of the committed code, which never set a seed.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def add_config_arguments(parser) -> None:
    """Attach the shared ``--config`` / Table III override flags to a parser.

    Every flag defaults to ``None`` so that "not passed" is distinguishable
    from "passed the default value", which is what lets the YAML win over
    the defaults but lose to an explicit command-line flag.
    """
    g = parser.add_argument_group(
        "phase configuration (Table III)",
        "Values resolve as: command-line flag > --config YAML > Table III default.",
    )
    g.add_argument("--config", default=None,
                   help="path to configs/phase*.yaml")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--lora-r", type=int, default=None)
    g.add_argument("--lora-alpha", type=int, default=None)
    g.add_argument("--lora-dropout", type=float, default=None)
    g.add_argument("--target-modules", default=None,
                   help="comma-separated, or 'legacy' for the pre-Table-III "
                        "seven-projection set")
    g.add_argument("--learning-rate", type=float, default=None)
    g.add_argument("--grad-accum", type=int, default=None)
    g.add_argument("--warmup-steps", type=int, default=None)
    g.add_argument("--lr-scheduler-type", default=None)
    g.add_argument("--optim", default=None)
    g.add_argument("--print-config", action="store_true",
                   help="print the resolved configuration and exit")


def config_from_args(args, **extra: Any) -> PhaseConfig:
    """Build a ``PhaseConfig`` from parsed args produced by the group above."""
    overrides: Dict[str, Any] = {"lora": {}, "training": {}}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    for src, dst in (("lora_r", "r"), ("lora_alpha", "alpha"),
                     ("lora_dropout", "dropout")):
        v = getattr(args, src, None)
        if v is not None:
            overrides["lora"][dst] = v
    tm = getattr(args, "target_modules", None)
    if tm:
        overrides["lora"]["target_modules"] = tm
    for src, dst in (("learning_rate", "learning_rate"),
                     ("grad_accum", "gradient_accumulation_steps"),
                     ("warmup_steps", "warmup_steps"),
                     ("lr_scheduler_type", "lr_scheduler_type"),
                     ("optim", "optim")):
        v = getattr(args, src, None)
        if v is not None:
            overrides["training"][dst] = v
    overrides = _deep_merge(overrides, extra or {})
    overrides = {k: v for k, v in overrides.items() if v not in ({}, None)}
    return PhaseConfig.load(getattr(args, "config", None), **overrides)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import argparse
    ap = argparse.ArgumentParser(description="Print a resolved phase config.")
    add_config_arguments(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    print(cfg.describe())
    print()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
