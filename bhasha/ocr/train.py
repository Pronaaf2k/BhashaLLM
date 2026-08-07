#!/usr/bin/env python3
"""
Phase 3 -- fine-tune swapnillo/Bangla-OCR-SFT (Qwen3-VL) with QLoRA.
Uses QLoRA for memory efficiency.

Paper Sec. IV-C describes this phase as trained on "the Ekush dataset [28]
plus the manually collected pages", with two documented deviations from
Table III: 2,000 optimiser steps rather than 500, and a learning rate of
1e-4 rather than 2e-4.

Training data -- resolved (was docs/ERRATA.md C1)
------------------------------------------------
The defaults below read ``data/processed/banglawriting`` and write
``models/ocr_adapters/banglawriting_adapter``. ``bhasha/eval/ocr_models.py``
scores that adapter against ``data/processed/banglawriting/test.jsonl``, and
``bhasha/scripts/train_ocr_improved.py`` combines Ekush with BanglaWriting.
The committed adapter was therefore produced from **BanglaWriting**, a
public handwritten Bangla dataset the manuscript does not cite, not from
Ekush plus self-collected pages as Sec. IV-C states. See ``docs/ERRATA.md``
C1 for the full record. The defaults are left unchanged so the existing
adapter remains reproducible; use ``--config configs/phase3_ocr_sft.yaml``
or ``--data_dir`` to train the composition Sec. IV-C describes.

Config integration (added; nothing removed)
-------------------------------------------
``--config configs/phase3_ocr_sft.yaml`` now works. Every original flag
(``--model_id``, ``--data_dir``, ``--output_dir``, ``--batch_size``,
``--grad_accum``, ``--epochs``, ``--lr``, ``--max_steps``, ``--save_steps``,
``--logging_steps``) behaves as before and still wins over the YAML.
``logs/phase3_summary.json`` is written on completion.

Note on LoRA rank, carried from paper Sec. III-C: r=16 was inherited from
the text phases and "no rank sweep was run to check whether 16 suffices for
the vision encoder".
"""
import json
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from pathlib import Path

from bhasha.data.dataset import OCRDataset, collate_fn
from bhasha.config import add_config_arguments, config_from_args, set_global_seed
from bhasha.utils.run_summary import RunSummary, count_trainable

from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed" / "banglawriting"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "ocr_adapters" / "banglawriting_adapter"

# Phase 3 deviations from Table III, both stated in paper Sec. IV-C.
PHASE3_DEFAULT_LR = 1e-4      # deviation: 2e-4 oscillated without settling
PHASE3_DEFAULT_STEPS = 2000   # deviation: image set is ~10x the text sets


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, default="swapnillo/Bangla-OCR-SFT")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=None)  # Table III: 1
    parser.add_argument("--grad_accum", type=int, default=None)  # Table III: 4
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)  # Sec. IV-C: 1e-4
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Square edge in px. configs/phase3 sets 256; "
                             "paper Sec. V-C: 224->256 costs 40ms, gains 0.02 "
                             "reported confidence.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Directory that relative image paths in the "
                             "JSONL resolve against.")
    parser.add_argument("--skip_missing_images", action="store_true",
                        help="Drop records whose image file is absent instead "
                             "of raising. The count is reported.")
    add_config_arguments(parser)
    args = parser.parse_args()

    # ---- resolve configuration: CLI flag > --config YAML > Table III ----
    cfg = config_from_args(args, base_model=args.model_id)
    cfg.training.setdefault("learning_rate", PHASE3_DEFAULT_LR)
    cfg.training.setdefault("max_steps", PHASE3_DEFAULT_STEPS)
    if args.batch_size is not None:
        cfg.training["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg.training["gradient_accumulation_steps"] = args.grad_accum
    if args.lr is not None:
        cfg.training["learning_rate"] = args.lr
    if args.max_steps and args.max_steps > 0:
        cfg.training["max_steps"] = args.max_steps
    if args.logging_steps:
        cfg.training["logging_steps"] = args.logging_steps
    if not cfg.phase or cfg.phase == "unnamed":
        cfg.phase = "3_ocr_sft"

    print(cfg.describe())
    if args.print_config:
        return cfg
    set_global_seed(cfg.seed)

    image_size = (
        args.image_size if args.image_size is not None
        else int(cfg.data.get("image_size", 256))
    )
    dataset_kwargs = dict(
        image_size=image_size,
        max_length=cfg.max_seq_length,
        image_root=args.image_root,
        skip_missing=args.skip_missing_images,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Check for train/val splits
    data_path = Path(args.data_dir)
    train_path = data_path / "train.jsonl"
    val_path = data_path / "val.jsonl"
    
    # Fallback to single file if splits don't exist (backward compatibility for Ekush)
    if not train_path.exists():
        print(f"Split files not found in {args.data_dir}. Checking for ocr_training_data.jsonl...")
        single_file = Path("/home/node/.openclaw/workspace/BhashaLLM/data/processed/ocr_training_data.jsonl")
        if single_file.exists():
            print(f"Using single file: {single_file}")
            full_dataset = OCRDataset(str(single_file), processor, **dataset_kwargs)
            # Create simple split
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        else:
            raise FileNotFoundError(f"No training data found in {args.data_dir} or standard paths.")
    else:
        print(f"Loading training data from: {train_path}")
        print(f"Loading validation data from: {val_path}")
        train_dataset = OCRDataset(str(train_path), processor, **dataset_kwargs)
        val_dataset = OCRDataset(str(val_path), processor, **dataset_kwargs)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")

    # Load model in 4-bit. Table III: NF4 with double quantisation. The
    # compute dtype stays bfloat16 here (the text phases use float16) because
    # the vision tower is loaded in bfloat16 below; mixing them silently
    # upcasts on every forward pass.
    bnb_kwargs = cfg.bnb_kwargs()
    bnb_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
    bnb_config = BitsAndBytesConfig(**bnb_kwargs)

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Lora Config. Table III: r=16, alpha=32, dropout 0.05, q_proj/v_proj.
    # Paper Sec. III-C notes the rank was inherited from the text phases and
    # never swept for the vision encoder. Pass --target-modules legacy for
    # the previous seven-projection set.
    lora_kwargs = cfg.lora_kwargs()
    lora_kwargs["modules_to_save"] = None
    peft_config = LoraConfig(**lora_kwargs)

    # Wrap model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    param_stats = count_trainable(model)

    # Training Arguments
    ta_kwargs = cfg.training_kwargs(args.output_dir)
    ta_kwargs.update({
        "weight_decay": cfg.training.get("weight_decay", 0.05),  # regularization
        "eval_strategy": "no",
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "load_best_model_at_end": False,
        "fp16": True,
        "bf16": False,
        # collate_fn returns provenance keys (image_id, writer_id) that are
        # not model inputs; Trainer must not try to prune them.
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    })
    if "max_steps" not in ta_kwargs and args.epochs:
        ta_kwargs["num_train_epochs"] = args.epochs
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    summary = RunSummary(
        cfg,
        n_train_sequences=len(train_dataset),
        n_eval_sequences=len(val_dataset),
        extra={
            "output_dir": str(args.output_dir),
            "data_dir": str(args.data_dir),
            "image_size": image_size,
            "images_skipped_missing": getattr(train_dataset, "n_skipped", 0),
            # Recorded on every run so the training-data provenance the
            # manuscript gets wrong is unambiguous in the log itself.
            "training_data_note":
                "docs/ERRATA.md C1: the committed adapter was trained on "
                "BanglaWriting, not the Ekush + self-collected composition "
                "described in paper Sec. IV-C. Check data_dir above against "
                "the dataset you intend to cite.",
            **param_stats,
        },
    ).start()

    print("Starting training...")
    trainer.train()

    print(f"Model saved to {args.output_dir}")
    trainer.save_model(args.output_dir)

    path = summary.finish(trainer).write()
    print(summary.report())
    print(f"wrote {path}")
    return cfg


if __name__ == "__main__":
    main()
