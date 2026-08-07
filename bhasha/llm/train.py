"""Phase 1 — continued pre-training on Bangla literary text (paper Sec. IV-C).

This is the original training script with the paper's Table III
configuration wired in. Nothing was removed: every command-line flag that
worked before still works, ``train(args)`` keeps its signature, and the
default output paths are unchanged.

What changed, and why
---------------------
1. ``--config configs/phase1_bangla_pt.yaml`` now works. ``README.md``
   documented that command and ``configs/`` was committed with the paper's
   hyperparameters, but no code read those files.

2. Defaults are Table III. Previously this script hardcoded LoRA over all
   seven projection matrices, ``paged_adamw_32bit``, no LR schedule, no
   seed, and a 1024-token context. Table III specifies ``q_proj, v_proj``
   (which is what yields Table II's 9.4M trainable parameters), AdamW,
   cosine decay after 50 warm-up steps, seed 42, and 512 tokens. Pass
   ``--target-modules legacy`` to restore the previous seven-projection
   behaviour exactly.

3. Sequence packing (``data.packing: true`` in the config) concatenates
   documents and chunks them to ``max_seq_length`` instead of padding each
   line to full length. Paper Sec. IV-C says "Sequences were packed to 512
   tokens"; the previous code padded, which wastes most of every batch on
   padding and makes the sequence count — and therefore ``epochs_covered``
   — mean something different from what the paper reports.

4. ``logs/phase1_summary.json`` is written on completion, carrying measured
   peak VRAM and a *computed* ``epochs_covered``. See
   ``bhasha/utils/run_summary.py`` and ``docs/ERRATA.md`` B2.

Usage
-----
    # paper-faithful, config-driven
    python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml

    # original-style invocation, still supported
    python -m bhasha.llm.train --model_name Qwen/Qwen2.5-1.5B-Instruct \\
        --max_steps 500

    # restore the pre-Table-III LoRA target set
    python -m bhasha.llm.train --config configs/phase1_bangla_pt.yaml \\
        --target-modules legacy
"""

import os
import json
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, Dataset

from pathlib import Path

from bhasha.config import (
    PhaseConfig, add_config_arguments, config_from_args, set_global_seed,
)
from bhasha.utils.run_summary import RunSummary, count_trainable

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "adapters"
LOG_DIR = BASE_DIR / "logs"


def _resolve(path_like):
    """Resolve a config path against the repository root."""
    p = Path(path_like)
    return p if p.is_absolute() else BASE_DIR / p


def pack_sequences(texts, tokenizer, block_size):
    """Concatenate documents and chunk them to exactly ``block_size`` tokens.

    Paper Sec. IV-C: "Sequences were packed to 512 tokens, giving about
    2,578 training sequences". Packing matters for more than throughput: the
    number of packed sequences is the denominator of ``epochs_covered``, so
    padding instead of packing changes the reported epoch coverage without
    changing anything about the run. An EOS token is inserted between
    documents so the model still sees a boundary.
    """
    eos = tokenizer.eos_token_id
    buffer, blocks = [], []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        buffer.extend(ids)
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]
    # The trailing partial block is dropped rather than padded, so every
    # sequence carries block_size real tokens and the count is exact.
    return blocks


def build_dataset(cfg, tokenizer, args, split):
    """Load one split, honouring the config's corpus paths when present.

    Falls back to the original ``data/processed/{split}.txt`` layout when
    the config does not name a corpus, so existing checkouts keep working.
    """
    key = {"train": "corpus", "val": "eval_corpus"}[split]
    configured = cfg.data.get(key)
    path = _resolve(configured) if configured else (DATA_DIR / f"{split}.txt")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{split} corpus not found at {path}. Generate the splits with:\n"
            f"  python -m bhasha.data.text_corpus --input <raw dirs> "
            f"--out-dir data/splits"
        )

    raw = load_dataset("text", data_files={"train": str(path)})["train"]
    block = cfg.max_seq_length if args.context_length is None else args.context_length

    if cfg.data.get("packing", True):
        blocks = pack_sequences(raw["text"], tokenizer, block)
        ds = Dataset.from_dict({"input_ids": blocks})
        return ds.map(
            lambda ex: {"attention_mask": [1] * len(ex["input_ids"])}
        )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=block,
            padding="max_length",
        )

    return raw.map(tokenize_function, batched=True, remove_columns=["text"])


def train(args):
    # ---- resolve configuration: CLI flag > --config YAML > Table III ----
    cfg = config_from_args(args, base_model=args.model_name)
    if args.batch_size is not None:
        cfg.training["per_device_train_batch_size"] = args.batch_size
    if args.context_length is not None:
        cfg.training["max_seq_length"] = args.context_length
    if args.max_steps is not None and args.max_steps > 0:
        cfg.training["max_steps"] = args.max_steps
    if not cfg.phase or cfg.phase == "unnamed":
        cfg.phase = "1_bangla_pt"

    print(cfg.describe())
    if args.print_config:
        return cfg
    set_global_seed(cfg.seed)

    print(f"Loading model: {cfg.base_model}")

    # 1. Quantization Config (Table III: 4-bit NF4, double quant)
    bnb_config = BitsAndBytesConfig(**cfg.bnb_kwargs())

    # 2. Load Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.get(
            "gradient_checkpointing", True)
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # usually right for training, left for generation

    # 3. LoRA Config (Table III: r=16, alpha=32, dropout=0.05, q_proj/v_proj)
    #    Target modules vary by architecture. Table III's q_proj/v_proj is
    #    what produces the 9.4M trainable parameters quoted in Table II.
    #    For BLOOM the equivalents are query_key_value, dense,
    #    dense_h_to_4h, dense_4h_to_h -- pass them via --target-modules.
    peft_config = LoraConfig(**cfg.lora_kwargs())

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    param_stats = count_trainable(model)

    # 4. Load & Tokenize Dataset
    print("Loading datasets...")
    tokenized_train = build_dataset(cfg, tokenizer, args, "train")
    tokenized_val = build_dataset(cfg, tokenizer, args, "val")
    print(f"Train sequences: {len(tokenized_train)}  "
          f"Val sequences: {len(tokenized_val)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Training Arguments
    run_name = f"bangla_adapt_{cfg.base_model.split('/')[-1]}"
    out_dir = os.path.join(OUTPUT_DIR, run_name)
    ta_kwargs = cfg.training_kwargs(out_dir)
    ta_kwargs.update({
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "fp16": True,
        "logging_dir": str(LOG_DIR),
        "ddp_find_unused_parameters": (
            False if torch.cuda.device_count() > 1 else None
        ),
    })
    # num_train_epochs only applies when max_steps was not set.
    if "max_steps" not in ta_kwargs and args.epochs:
        ta_kwargs["num_train_epochs"] = args.epochs
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    summary = RunSummary(
        cfg,
        n_train_sequences=len(tokenized_train),
        n_eval_sequences=len(tokenized_val),
        extra={"run_name": run_name, "output_dir": out_dir, **param_stats},
    ).start()

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(os.path.join(out_dir, "final_adapter"))

    # Also save loss log (unchanged from the original script)
    log_history = trainer.state.log_history
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{run_name}_history.json"), 'w') as f:
        json.dump(log_history, f)

    # Phase summary: the evidence file docs/TRACEABILITY.md points at for
    # Table II's peak VRAM and every Table IV row.
    path = summary.finish(trainer).write()
    print(summary.report())
    print(f"wrote {path}")
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model name (paper Sec. III-B: "
                             "Qwen-2.5-1.5B-Instruct)")
    # Defaults are None so that "flag not passed" stays distinguishable from
    # "flag passed its default", which is what lets --config win.
    parser.add_argument("--batch_size", type=int, default=None)  # Table III: 1
    parser.add_argument("--context_length", type=int, default=None)  # Table III: 512
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)  # Table III: 500
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    add_config_arguments(parser)
    args = parser.parse_args()

    train(args)
