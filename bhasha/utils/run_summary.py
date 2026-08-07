"""Per-phase training summaries: peak VRAM, computed epoch coverage, losses.

``README.md`` prints a code block that writes ``logs/phase1_summary.json``
with ``peak_vram_gb`` and a *computed* ``epochs_covered``, and
``docs/TRACEABILITY.md`` names ``logs/phase*_summary.json`` as the evidence
file for Table II's 9.4 GB peak and every Table IV row. No such code was in
the repository, so those rows were unbacked. This module is that code,
factored so all three phases emit the same schema.

Why ``epochs_covered`` is computed and never typed
--------------------------------------------------
``docs/ERRATA.md`` B2: Sec. IV-C states the corpus packs into ~2,578
sequences and that 500 steps at effective batch 4 covers "roughly 0.8 of an
epoch". Those do not reconcile with Table IV's 5.28M-token training split —
5.28e6 / 512 = 10,312 sequences, and 500 x 4 = 2,000 sequences is 0.19
epochs, not 0.8. ``configs/phase1_bangla_pt.yaml`` carries the same note and
instructs that the value be computed. Whatever this module prints is the
figure to cite.

Why peak VRAM is recorded here rather than observed by hand
-----------------------------------------------------------
Table II reports 9.4 GB peak for the QLoRA row and ``docs/ERRATA.md`` B1
records that no 11B training run exists to back it, so the figure is an
estimate from a feasibility analysis. Any run through this module produces
a measured number for the model actually trained, which is the honest
replacement.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _torch():
    try:
        import torch
        return torch
    except ImportError:  # pragma: no cover
        return None


def gpu_info() -> Dict[str, Any]:
    """Device name, capability and total memory, or a reason it is absent."""
    torch = _torch()
    if torch is None:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda not available"}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_memory_gb": round(props.total_memory / 1e9, 3),
        "device_count": torch.cuda.device_count(),
    }


def environment() -> Dict[str, Any]:
    """Library versions, captured rather than transcribed.

    ``docs/ERRATA.md`` A1 records that the versions printed in paper
    Sec. IV-A describe an environment in which the project could not have
    run. Every summary this module writes carries the real ones.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name, mod in (("torch", "torch"), ("transformers", "transformers"),
                      ("peft", "peft"), ("bitsandbytes", "bitsandbytes"),
                      ("trl", "trl"), ("datasets", "datasets"),
                      ("accelerate", "accelerate")):
        try:
            env[name] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[name] = None
    torch = _torch()
    if torch is not None:
        env["cuda_runtime"] = getattr(torch.version, "cuda", None)
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    return env


def epochs_covered(
    optimizer_steps: int, effective_batch: int, n_train_sequences: Optional[int]
) -> Optional[float]:
    """Fraction of the training set actually seen.

    ``optimizer_steps * effective_batch`` is the number of *sequences*
    consumed, not the number of forward passes, which is the distinction
    that makes the Sec. IV-C arithmetic go wrong when done by hand.
    """
    if not n_train_sequences:
        return None
    return round(optimizer_steps * effective_batch / n_train_sequences, 4)


class RunSummary:
    """Collects everything ``logs/phase*_summary.json`` should contain.

    Usage::

        summary = RunSummary(cfg, n_train_sequences=len(train_ds))
        summary.start()
        trainer.train()
        summary.finish(trainer)
        summary.write()

    Safe to use without CUDA: the VRAM fields become ``None`` rather than
    raising, so the pipeline is testable on CPU.
    """

    def __init__(
        self,
        config,
        n_train_sequences: Optional[int] = None,
        n_eval_sequences: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.n_train_sequences = n_train_sequences
        self.n_eval_sequences = n_eval_sequences
        self.extra = dict(extra or {})
        self._t0: Optional[float] = None
        self.payload: Dict[str, Any] = {}

    def start(self) -> "RunSummary":
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self._t0 = time.time()
        return self

    # ------------------------------------------------------------- finish

    @staticmethod
    def _final_losses(log_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        train = [e["loss"] for e in log_history if "loss" in e]
        val = [e["eval_loss"] for e in log_history if "eval_loss" in e]
        return {
            "final_train_loss": round(train[-1], 6) if train else None,
            "final_eval_loss": round(val[-1], 6) if val else None,
            "best_eval_loss": round(min(val), 6) if val else None,
            "n_logged_train_points": len(train),
            "n_logged_eval_points": len(val),
        }

    def finish(self, trainer: Any = None, train_result: Any = None) -> "RunSummary":
        torch = _torch()
        peak = None
        reserved = None
        if torch is not None and torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            reserved = round(torch.cuda.max_memory_reserved() / 1e9, 3)

        log_history: List[Dict[str, Any]] = []
        steps = 0
        if trainer is not None and getattr(trainer, "state", None) is not None:
            log_history = list(trainer.state.log_history or [])
            steps = int(trainer.state.global_step or 0)
        if train_result is not None and not steps:
            steps = int(getattr(train_result, "global_step", 0) or 0)

        cfg = self.config
        eff = cfg.effective_batch
        self.payload = {
            "phase": cfg.phase,
            "base_model": cfg.base_model,
            "init_adapter": cfg.init_adapter,
            "seed": cfg.seed,
            "config_source": cfg.source_path or "Table III defaults",
            "steps": steps,
            "effective_batch": eff,
            "tokens_per_sequence": cfg.max_seq_length,
            "n_train_sequences": self.n_train_sequences,
            "n_eval_sequences": self.n_eval_sequences,
            # Computed, never typed. See module docstring and ERRATA B2.
            "epochs_covered": epochs_covered(steps, eff, self.n_train_sequences),
            "epochs_covered_note":
                "Computed as steps * effective_batch / n_train_sequences. "
                "Paper Sec. IV-C and Table IV report 0.8 for Phase 1, which "
                "does not reconcile with the stated corpus size "
                "(docs/ERRATA.md B2). Cite this field, not the paper.",
            "sequences_consumed": steps * eff,
            "peak_vram_gb": peak,
            "peak_vram_reserved_gb": reserved,
            "peak_vram_note":
                "Measured with torch.cuda.max_memory_allocated for the model "
                "actually trained. Table II's 9.4 GB is an estimate from a "
                "feasibility analysis, not a measurement (docs/ERRATA.md B1).",
            "wall_clock_s": round(time.time() - self._t0, 1) if self._t0 else None,
            "gpu": gpu_info(),
            "environment": environment(),
            "hyperparameters": {
                "quantization": cfg.quantization,
                "lora": cfg.lora_kwargs(),
                "training": cfg.training,
            },
            **self._final_losses(log_history),
            **self.extra,
        }
        if log_history:
            self.payload["log_history"] = log_history
        return self

    # -------------------------------------------------------------- output

    def write(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path else self.config.summary_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return target

    def report(self) -> str:
        p = self.payload
        lines = [
            "",
            "=" * 66,
            f"  phase              {p.get('phase')}",
            f"  seed               {p.get('seed')}",
            f"  steps              {p.get('steps')}  (effective batch "
            f"{p.get('effective_batch')})",
            f"  sequences seen     {p.get('sequences_consumed')} of "
            f"{p.get('n_train_sequences')}",
            f"  epochs_covered     {p.get('epochs_covered')}   <- computed, "
            f"see ERRATA B2",
            f"  peak VRAM          {p.get('peak_vram_gb')} GB allocated / "
            f"{p.get('peak_vram_reserved_gb')} GB reserved",
            f"  final train loss   {p.get('final_train_loss')}",
            f"  final eval loss    {p.get('final_eval_loss')}",
            f"  wall clock         {p.get('wall_clock_s')} s",
            "=" * 66,
        ]
        return "\n".join(lines)


def count_trainable(model) -> Dict[str, Any]:
    """Trainable vs total parameters, for the Table II comparison.

    Table II attributes 9.4M trainable parameters to the ``r=16`` QLoRA
    configuration. That figure only holds for ``target_modules = q_proj,
    v_proj``; the seven-projection set the repository previously used gives
    several times more. Logging this makes the discrepancy visible in the
    run summary instead of invisible in the code.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else None,
        "table_ii_reference_trainable": 9_400_000,
    }
