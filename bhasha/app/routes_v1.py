"""Paper-faithful API surface, mounted alongside the existing endpoints.

``bhasha/app/api.py`` exposes ``/api/analyze``, ``/api/chat`` and
``/api/philosophical``, backed by a ResNet-34 three-head grapheme
classifier and the Gemini cloud API. Neither appears anywhere in the
manuscript, and the Gemini dependency sits awkwardly beside Sec. III-F and
Sec. VI-F, which describe a fully offline deployment. Those endpoints are
**not modified or removed** -- they are working functionality and remain
mounted exactly as they were.

This router adds the surface the paper actually describes:

===============================  ==============================================
``GET  /api/v1/status``          resident models, active adapter, VRAM
``POST /api/v1/generate``        Bangla text generation (Sec. IV-C decoding)
``POST /api/v1/grade``           short-answer grading with Bangla feedback
``POST /api/v1/ocr``             handwritten Bangla transcription
``POST /api/v1/adapter``         explicit adapter swap (Sec. III-F)
===============================  ==============================================

Everything runs locally through ``bhasha.app.adapter_manager``: base model
loaded once, at most one adapter resident, no network calls.

Models are loaded lazily on first use, so importing this router costs
nothing and the server starts without a GPU. A request that needs a missing
adapter returns 503 with the path it looked for, rather than a stack trace.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from bhasha.app.adapter_manager import (
    OCR_DECODING, OCR_IMAGE_SIZE, TEXT_DECODING, get_manager,
)

router = APIRouter(prefix="/api/v1", tags=["bhasha-v1"])


# ------------------------------------------------------------------ schemas


class GenerateRequest(BaseModel):
    prompt: str
    # Paper Sec. III-F names two runtime adapters. None means the bare base
    # model, which is the "Before" configuration of Table VII.
    adapter: Optional[str] = Field(
        default=None, description="bangla | grading | ocr | null for base")
    max_new_tokens: int = Field(default=TEXT_DECODING["max_new_tokens"])
    temperature: float = Field(default=TEXT_DECODING["temperature"])
    top_p: float = Field(default=TEXT_DECODING["top_p"])
    repetition_penalty: float = Field(default=TEXT_DECODING["repetition_penalty"])
    return_logprobs: bool = Field(
        default=False,
        description="return per-token logprobs for eval/confidence.py")
    chat: bool = Field(
        default=True,
        description="apply the model's ChatML template (paper Sec. III-F)")


class GradeRequest(BaseModel):
    question: str
    reference_answer: str
    student_answer: str
    max_new_tokens: int = Field(default=TEXT_DECODING["max_new_tokens"])


class AdapterRequest(BaseModel):
    adapter: Optional[str] = Field(
        default=None, description="null unloads every adapter")


# ------------------------------------------------------------------- routes


def _manager():
    return get_manager()


def _handle(exc: Exception) -> HTTPException:
    """Map manager errors to useful status codes.

    A missing adapter is an operational condition, not a bug: the adapters
    are 3.6 GB and are not in git (README, "Models"). 503 with the path it
    looked for is more useful than a 500 with a traceback.
    """
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Resident models, active adapter and VRAM.

    Evidence for the Sec. III-F residency claim: `vram_allocated_gb` should
    not grow across adapter swaps, because the outgoing adapter is deleted
    before the incoming one is loaded.
    """
    m = _manager()
    return {
        "status": "ok",
        "paper_section": "III-F",
        **m.status(),
        "decoding_defaults": {"text": TEXT_DECODING, "ocr": OCR_DECODING},
        "offline": True,
        "note":
            "This router makes no network calls. The legacy /api/chat and "
            "/api/philosophical endpoints proxy to the Gemini cloud API and "
            "are not part of the pipeline described in the paper.",
    }


@router.post("/adapter")
async def swap_adapter(req: AdapterRequest) -> Dict[str, Any]:
    """Swap the resident adapter explicitly and report the VRAM delta."""
    try:
        rec = _manager().use(req.adapter)
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc
    return {
        "from": rec.from_adapter,
        "to": rec.to_adapter,
        "vram_before_gb": rec.vram_before_gb,
        "vram_after_gb": rec.vram_after_gb,
        "seconds": rec.seconds,
    }


@router.post("/generate")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    """Bangla text generation under paper Sec. IV-C's fixed decoding."""
    m = _manager()
    try:
        prompt = (
            m.format_chatml([{"role": "user", "content": req.prompt}])
            if req.chat else req.prompt
        )
        return m.generate(
            prompt,
            task=req.adapter,
            return_logprobs=req.return_logprobs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.post("/grade")
async def grade(req: GradeRequest) -> Dict[str, Any]:
    """Short-answer grading with Bangla feedback (paper Sec. III-B, Phase 2).

    The prompt is Bangla-only, per Sec. IV-C. The grading adapter is loaded
    automatically; the caller does not need to know the input format, which
    is the property Sec. III-F asks of the primary interface.
    """
    try:
        return _manager().grade(
            req.question, req.reference_answer, req.student_answer,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc


@router.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    image_size: int = OCR_IMAGE_SIZE,
    return_confidence: bool = True,
    use_adapter: bool = True,
) -> Dict[str, Any]:
    """Transcribe handwritten Bangla with the Phase-3 QLoRA recogniser.

    Resizing and normalisation match ``bhasha/data/dataset.py`` exactly, so
    inference preprocessing cannot drift from training preprocessing.
    ``return_confidence`` computes the ``eval/confidence.py`` definition
    (exp mean token logprob) rather than an undefined quantity -- see
    ``docs/ERRATA.md`` B11.

    ``use_adapter=false`` runs the un-adapted base model, which is Table
    VII's "Before" column (28% CER, 0.68 confidence).
    """
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        from PIL import Image
        data = await image.read()
        pil = Image.open(io.BytesIO(data))
        return _manager().ocr(
            pil, image_size=image_size, return_logprobs=return_confidence,
            use_adapter=use_adapter)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _handle(exc) from exc
