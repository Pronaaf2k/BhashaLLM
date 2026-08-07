"""Phase 2 — instruction tuning for short-answer grading (paper Sec. IV-C).

Table IV: 400 train / 100 val / 50 held-out pairs, 5 epochs, 500 steps,
validation loss 0.018.

Config integration (added; nothing removed)
-------------------------------------------
``--config configs/phase2_grading_sft.yaml`` now works, so the Table III
hyperparameters and the ``data.train`` / ``data.validation`` /
``data.heldout`` paths in that file are honoured. Every original flag
(``--base_model``, ``--adapter_path``, ``--batch_size``, ``--epochs``,
``--context_length``, ``--resume_from_checkpoint``) behaves as before.

Two behaviours changed to match the paper, both overridable:

* When the config names ``data.train`` and ``data.validation``, those files
  are used instead of a random 90/10 split of a single file. Paper Sec. IV-C
  describes a *fixed* 400/100 split with a further 50 pairs "written
  afterwards and never used for tuning". A fresh random split on every run
  cannot honour that, and silently leaks held-out pairs into training
  across runs.
* The LoRA fallback path (used only when no adapter is supplied) now
  targets Table III's ``q_proj, v_proj``. Pass ``--target-modules legacy``
  for the previous seven-projection set.
"""

import os
import torch
import argparse
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configuration
from pathlib import Path

from bhasha.config import add_config_arguments, config_from_args, set_global_seed
from bhasha.utils.run_summary import RunSummary, count_trainable

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = str(BASE_DIR / "models" / "instruct_adapters")
LOG_DIR = str(BASE_DIR / "logs")
DATA_PATH = str(BASE_DIR / "data" / "processed" / "kaggle_instruct.jsonl")


def _resolve(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else BASE_DIR / p


def load_grading_splits(cfg, args):
    """Return (train_ds, eval_ds) honouring the fixed split when configured.

    Paper Sec. IV-C: 400 training pairs and 100 validation pairs, fixed.
    Falls back to the original random 90/10 split of ``DATA_PATH`` when the
    config does not name split files, so existing checkouts keep working.
    """
    train_path = cfg.data.get("train")
    val_path = cfg.data.get("validation")
    if train_path and val_path and _resolve(train_path).exists() \
            and _resolve(val_path).exists():
        print(f"Using fixed splits from config: {train_path} / {val_path}")
        train_ds = load_dataset("json", data_files=str(_resolve(train_path)),
                                split="train")
        eval_ds = load_dataset("json", data_files=str(_resolve(val_path)),
                               split="train")
        held = cfg.data.get("heldout")
        if held:
            print(f"Held-out set (never used for tuning): {held}")
        return train_ds, eval_ds

    data_path = str(_resolve(train_path)) if train_path else DATA_PATH
    print(f"Loading dataset from {data_path}...")
    print("NOTE: no fixed val split configured; falling back to a random "
          "90/10 split. Paper Sec. IV-C describes a fixed 400/100 split -- "
          "set data.train and data.validation in the phase config to honour it.")
    dataset = load_dataset('json', data_files=data_path, split='train')
    parts = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    return parts["train"], parts["test"]


def train(args):
    cfg = config_from_args(args, base_model=args.base_model)
    if args.batch_size is not None:
        cfg.training["per_device_train_batch_size"] = args.batch_size
    if args.context_length is not None:
        cfg.training["max_seq_length"] = args.context_length
    if args.epochs is not None:
        cfg.training["num_train_epochs"] = args.epochs
    if args.adapter_path:
        cfg.init_adapter = args.adapter_path
    if not cfg.phase or cfg.phase == "unnamed":
        cfg.phase = "2_grading_sft"

    print(cfg.describe())
    if args.print_config:
        return cfg
    set_global_seed(cfg.seed)

    print(f"Loading base model: {cfg.base_model}")

    bnb_config = BitsAndBytesConfig(**cfg.bnb_kwargs())

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    # Load adapter for continued training. Paper Sec. IV-C: "Instruction
    # tuning for grading used the same base model with the Bangla adapter
    # loaded", i.e. Phase 2 continues from the Phase 1 adapter.
    adapter_path = cfg.init_adapter
    if adapter_path:
        print(f"Loading adapter for continued training from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(_resolve(adapter_path)),
                                          is_trainable=True)
        # Ensure we are in training mode
        model.print_trainable_parameters()
    else:
        # Should not happen in this phase, but fallback
        print("No adapter path provided. Initializing new LoRA config.")
        model = prepare_model_for_kbit_training(model)
        # Table III target modules. --target-modules legacy restores the
        # previous seven-projection set.
        peft_config = LoraConfig(**cfg.lora_kwargs())
        model = get_peft_model(model, peft_config)

    param_stats = count_trainable(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, eval_ds = load_grading_splits(cfg, args)

    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    print("Starting SFT...")
    sft_kwargs = cfg.training_kwargs(OUTPUT_DIR)
    # SFTConfig names the sequence cap `max_length`, not `max_seq_length`.
    sft_kwargs.update({
        "max_length": cfg.max_seq_length,
        "dataset_text_field": cfg.data.get("text_field", "text"),
        "bf16": True,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        # Keep more checkpoints so past progress isn't overwritten easily
        "save_total_limit": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    })
    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # No peft_config here, as model is already a PeftModel
    )

    summary = RunSummary(
        cfg,
        n_train_sequences=len(train_ds),
        n_eval_sequences=len(eval_ds),
        extra={"output_dir": OUTPUT_DIR, **param_stats},
    ).start()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_instruct_adapter"))
    print(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_instruct_adapter')}")

    path = summary.finish(trainer).write()
    print(summary.report())
    print(f"wrote {path}")
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No longer `required`: --config supplies base_model for the documented
    # `--config configs/phase2_grading_sft.yaml` invocation. Passing
    # --base_model explicitly still works and still wins.
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)  # Table III: 1
    parser.add_argument("--epochs", type=int, default=None)      # Table IV: 5
    parser.add_argument("--context_length", type=int, default=None)  # Table III: 512
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint")
    add_config_arguments(parser)
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    train(args)
