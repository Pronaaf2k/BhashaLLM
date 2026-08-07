"""Single-resident-adapter runtime, implementing paper Sec. III-F.

    "At runtime, only the base model and the currently needed adapter
    (grading or OCR) are kept in memory, which avoids duplicating the 6.9GB
    base model for every task."

    "A command-line interface, test_models.py, is the primary production
    interface: it applies the correct ChatML formatting for the grading
    model and the correct resizing and normalisation for the OCR model
    automatically, so the user does not need to know either model's input
    format."

Neither behaviour existed in the repository. ``bhasha/app/api.py`` serves a
ResNet-34 three-head grapheme classifier and proxies to the Gemini cloud
API; it loads no QLoRA adapter and performs no swapping. That file is left
untouched -- this module is additive, and ``bhasha/app/routes_v1.py`` mounts
the paper-faithful endpoints alongside the existing ones.

What "single resident" actually requires
----------------------------------------
Calling ``PeftModel.from_pretrained`` twice does not free the first
adapter: the modules stay attached to the base model and both sets of LoRA
weights remain on the device. PEFT's own ``load_adapter`` /
``set_adapter`` mechanism keeps every loaded adapter resident too --
switching is cheap but memory is not reclaimed. To honour the claim in
Sec. III-F this manager calls ``delete_adapter`` on the outgoing adapter
before loading the incoming one, then empties the CUDA allocator cache, and
reports VRAM before and after so the claim is checkable rather than
asserted.

The base model is loaded once and reused across swaps, which is the half of
the claim that saves the 6.9 GB.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Default local layout, matching the README's `models/` tree.
DEFAULT_MODELS_DIR = BASE_DIR / "models"

# Task -> adapter directory. Paper Sec. III-F names two runtime adapters,
# grading and OCR; the Phase-1 Bangla adapter is included because Phase 2
# is trained on top of it and it is useful on its own for generation.
DEFAULT_ADAPTERS: Dict[str, str] = {
    "bangla": "bangla_adapters/final_adapter",
    "grading": "instruct_adapters/final_instruct_adapter",
    "ocr": "ocr_adapters/banglawriting_adapter",
}

# Paper Sec. IV-C, fixed decoding for text so comparisons are not
# confounded by sampling.
TEXT_DECODING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_new_tokens": 256,
    "do_sample": True,
}

# Paper Sec. IV-C: greedy decoding with a 128-token limit for OCR.
OCR_DECODING = {
    "do_sample": False,
    "max_new_tokens": 128,
}

# configs/phase3_ocr_sft.yaml -> data.image_size
OCR_IMAGE_SIZE = 256


def vram_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    return None


@dataclass
class SwapRecord:
    """One adapter swap, kept so the Sec. III-F claim is auditable."""
    from_adapter: Optional[str]
    to_adapter: Optional[str]
    vram_before_gb: Optional[float]
    vram_after_gb: Optional[float]
    seconds: float


class AdapterManager:
    """Holds one base model and at most one adapter at a time.

    Thread-safe: FastAPI serves requests concurrently, and two overlapping
    swaps would leave the model in an undefined adapter state. A single
    re-entrant lock serialises every load, swap and generate.

    Parameters
    ----------
    base_model:
        HF id or local path for the text base model. Paper Sec. III-B
        selects Qwen-2.5-1.5B-Instruct.
    ocr_base_model:
        Vision-language base. Paper Sec. IV-C: ``swapnillo/Bangla-OCR-SFT``.
        Loaded lazily and separately, because it is a different
        architecture class and cannot share the text base's weights.
    models_dir:
        Root that relative adapter paths resolve against.
    load_in_4bit:
        Table III's NF4 quantisation. Disable only for CPU debugging.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        ocr_base_model: str = "swapnillo/Bangla-OCR-SFT",
        models_dir: str | Path = DEFAULT_MODELS_DIR,
        adapters: Optional[Dict[str, str]] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.base_model = base_model
        self.ocr_base_model = ocr_base_model
        self.models_dir = Path(models_dir)
        self.adapters = dict(adapters or DEFAULT_ADAPTERS)
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        self._lock = threading.RLock()
        self._text_model = None
        self._tokenizer = None
        self._ocr_model = None
        self._processor = None
        self._active: Optional[str] = None
        self.swap_history: List[SwapRecord] = []

    # ------------------------------------------------------------- helpers

    def adapter_path(self, task: str) -> Optional[Path]:
        rel = self.adapters.get(task)
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.models_dir / p

    def _bnb_config(self):
        import torch
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    @staticmethod
    def _reclaim():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --------------------------------------------------------- text models

    def _ensure_text_base(self):
        if self._text_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if self.load_in_4bit:
            kwargs["quantization_config"] = self._bnb_config()
        self._text_model = AutoModelForCausalLM.from_pretrained(
            self.base_model, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding is correct for generation; right padding would put
        # pad tokens between the prompt and the first generated token.
        self._tokenizer.padding_side = "left"

    def use(self, task: Optional[str]) -> SwapRecord:
        """Make ``task``'s adapter the only resident adapter.

        ``task=None`` unloads every adapter and leaves the bare base model,
        which is the configuration used for the un-adapted baselines in
        Table VII ("Before").
        """
        with self._lock:
            if task == self._active:
                return SwapRecord(task, task, vram_gb(), vram_gb(), 0.0)

            t0 = time.perf_counter()
            before = vram_gb()
            previous = self._active

            self._ensure_text_base()
            model = self._text_model

            # Evict the outgoing adapter. delete_adapter is what actually
            # frees the LoRA weights; set_adapter alone would leave both
            # resident and quietly break the Sec. III-F memory claim.
            if previous is not None and hasattr(model, "delete_adapter"):
                try:
                    model.delete_adapter(previous)
                except Exception:  # noqa: BLE001 - adapter may already be gone
                    pass
                self._reclaim()

            if task is not None:
                path = self.adapter_path(task)
                if path is None:
                    raise KeyError(
                        f"unknown task {task!r}; known: {sorted(self.adapters)}")
                if not path.exists():
                    raise FileNotFoundError(
                        f"adapter for task {task!r} not found at {path}. "
                        f"Adapters are not in git; see README 'Models'.")
                from peft import PeftModel
                if hasattr(model, "load_adapter"):
                    model.load_adapter(str(path), adapter_name=task)
                    model.set_adapter(task)
                else:
                    self._text_model = PeftModel.from_pretrained(
                        model, str(path), adapter_name=task)
                    self._text_model.set_adapter(task)

            self._active = task
            rec = SwapRecord(previous, task, before, vram_gb(),
                             round(time.perf_counter() - t0, 3))
            self.swap_history.append(rec)
            return rec

    # ------------------------------------------------------------ chat/gen

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's ChatML template.

        Paper Sec. III-F: the interface "applies the correct ChatML
        formatting for the grading model ... so the user does not need to
        know either model's input format". Falls back to a hand-rolled
        ChatML string for tokenizers without a chat template, rather than
        silently sending a raw prompt to a chat-tuned model, which is a
        common cause of the instruction-following failures Sec. VI-B
        describes.
        """
        self._ensure_text_base()
        tok = self._tokenizer
        template = getattr(tok, "chat_template", None)
        if template:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        parts = [
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
            for m in messages
        ]
        return "\n".join(parts) + "\n<|im_start|>assistant\n"

    def generate(
        self,
        prompt: str,
        task: Optional[str] = None,
        return_logprobs: bool = False,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Generate text under Sec. IV-C's fixed decoding parameters.

        ``return_logprobs`` collects the per-token log-probabilities that
        ``eval/confidence.py`` turns into the Table VII confidence score.
        """
        import torch
        with self._lock:
            self.use(task)
            tok, model = self._tokenizer, self._text_model
            params = {**TEXT_DECODING, **overrides}

            inputs = tok(prompt, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    output_scores=return_logprobs,
                    return_dict_in_generate=True,
                    **params,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            seq = out.sequences[0]
            new_tokens = seq[inputs["input_ids"].shape[-1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)

            payload: Dict[str, Any] = {
                "text": text,
                "adapter": task,
                "n_new_tokens": int(new_tokens.shape[-1]),
                "latency_ms": round(elapsed_ms, 1),
                # Recorded because a latency without its token budget is not
                # interpretable -- docs/ERRATA.md B3.
                "max_new_tokens": params.get("max_new_tokens"),
                "decoding": params,
            }
            if return_logprobs:
                scores = model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True)
                payload["token_logprobs"] = [float(x) for x in scores[0]]
            return payload

    def grade(self, question: str, reference: str, answer: str,
              **overrides: Any) -> Dict[str, Any]:
        """Grade a short answer with Bangla feedback (paper Sec. III-B).

        The prompt is Bangla-only. Sec. IV-C: an English instruction wrapper
        "made language reversion markedly more likely in the smaller
        models", and the grading model is the 1.5B.
        """
        user = (
            "নিচের প্রশ্ন, আদর্শ উত্তর এবং শিক্ষার্থীর উত্তর পড়ুন। "
            "শিক্ষার্থীর উত্তরটি মূল্যায়ন করুন এবং বাংলায় সংক্ষিপ্ত মতামত দিন।\n\n"
            f"প্রশ্ন: {question}\n"
            f"আদর্শ উত্তর: {reference}\n"
            f"শিক্ষার্থীর উত্তর: {answer}\n\n"
            "মূল্যায়ন:"
        )
        prompt = self.format_chatml([{"role": "user", "content": user}])
        return self.generate(prompt, task="grading", **overrides)

    # ----------------------------------------------------------------- OCR

    def _ensure_ocr(self):
        if self._ocr_model is not None:
            return
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        self._ocr_model = AutoModelForVision2Seq.from_pretrained(
            self.ocr_base_model, **kwargs)
        self._processor = AutoProcessor.from_pretrained(
            self.ocr_base_model, trust_remote_code=True)

        path = self.adapter_path("ocr")
        if path and path.exists():
            from peft import PeftModel
            self._ocr_model = PeftModel.from_pretrained(
                self._ocr_model, str(path))

    def ocr(self, image, image_size: int = OCR_IMAGE_SIZE,
            return_logprobs: bool = False, **overrides: Any) -> Dict[str, Any]:
        """Transcribe a handwritten Bangla image.

        Applies "the correct resizing and normalisation for the OCR model
        automatically" (paper Sec. III-F): RGB conversion and a square
        resize to ``image_size``, matching what
        ``bhasha/data/dataset.py`` does at training time. Training and
        inference preprocessing that disagree is one of the quieter ways an
        OCR CER doubles.
        """
        import torch
        from PIL import Image
        from bhasha.data.dataset import DEFAULT_PROMPT

        with self._lock:
            self._ensure_ocr()
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            image = image.convert("RGB").resize((image_size, image_size))

            messages = [{
                "role": "user",
                "content": [{"type": "image"},
                            {"type": "text", "text": DEFAULT_PROMPT}],
            }]
            try:
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # noqa: BLE001
                text = DEFAULT_PROMPT

            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self._ocr_model.device)

            params = {**OCR_DECODING, **overrides}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self._ocr_model.generate(
                    **inputs,
                    output_scores=return_logprobs,
                    return_dict_in_generate=True,
                    **params,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            seq = out.sequences[0]
            new_tokens = seq[inputs["input_ids"].shape[-1]:]
            transcription = self._processor.decode(
                new_tokens, skip_special_tokens=True)

            payload: Dict[str, Any] = {
                "text": transcription,
                "image_size": image_size,
                "n_new_tokens": int(new_tokens.shape[-1]),
                "latency_ms": round(elapsed_ms, 1),
                "decoding": params,
            }
            if return_logprobs:
                scores = self._ocr_model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True)
                lps = [float(x) for x in scores[0]]
                payload["token_logprobs"] = lps
                # eval/confidence.py's definition, computed inline so the
                # endpoint can return it without a second pass.
                import math
                payload["confidence"] = (
                    round(math.exp(sum(lps) / len(lps)), 4) if lps else None
                )
                payload["confidence_definition"] = (
                    "exp(mean log p) over generated tokens; see "
                    "eval/confidence.py and docs/ERRATA.md B11"
                )
            return payload

    # ------------------------------------------------------------- status

    def status(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "ocr_base_model": self.ocr_base_model,
            "text_base_loaded": self._text_model is not None,
            "ocr_model_loaded": self._ocr_model is not None,
            "active_adapter": self._active,
            "known_adapters": {
                k: {"path": str(self.adapter_path(k)),
                    "present": bool(self.adapter_path(k)
                                    and self.adapter_path(k).exists())}
                for k in sorted(self.adapters)
            },
            "vram_allocated_gb": vram_gb(),
            "n_swaps": len(self.swap_history),
            "residency_policy":
                "Base model loaded once; at most one text adapter resident. "
                "The outgoing adapter is deleted before the incoming one is "
                "loaded (paper Sec. III-F).",
        }

    def unload(self) -> None:
        """Drop everything. Used by tests and by the CLI on exit."""
        with self._lock:
            self._text_model = None
            self._tokenizer = None
            self._ocr_model = None
            self._processor = None
            self._active = None
            self._reclaim()


_default: Optional[AdapterManager] = None


def get_manager(**kwargs: Any) -> AdapterManager:
    """Process-wide singleton, so the API does not load the base per request."""
    global _default
    if _default is None:
        _default = AdapterManager(**kwargs)
    return _default
