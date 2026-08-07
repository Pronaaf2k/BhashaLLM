"""Opt-in web frontend route — paper Sec. III-F.

    "A web frontend was added later as an optional interface for users who
    prefer not to use a terminal; it talks to the same backend and is not
    loaded unless explicitly opened, so it does not compete with the
    language models for GPU memory."

The repository contained no frontend of any kind: no HTML, no JavaScript,
no template directory, no static mount, no ``package.json``. The claim had
no artifact behind it. See ``docs/ERRATA.md`` C6.

Design follows the sentence above rather than general web practice:

**"not loaded unless explicitly opened"** is implemented as an *opt-in*
mount. ``python main.py`` serves the API alone; the UI appears only when
``BHASHA_ENABLE_UI=1`` is set or ``mount_ui(app, force=True)`` is called.
Mounting it by default would contradict the claim, and a research artifact
that quietly serves a web page on every start is also a small security
surprise.

**"talks to the same backend"** is implemented by giving the page no
backend of its own. Every action in ``static/index.html`` is a ``fetch``
against ``/api/v1/*`` — the same endpoints ``test_models.py`` drives. There
is no second inference path to keep in sync, and no way for the UI to
disagree with the CLI about how a model is prompted.

**"does not compete with the language models for GPU memory"** is
implemented by the page being a single static file with no build step, no
npm, no CDN and no framework. Serving it costs a file handle.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX = STATIC_DIR / "index.html"

# Set BHASHA_ENABLE_UI=1 to serve the frontend.
ENV_FLAG = "BHASHA_ENABLE_UI"

router = APIRouter(tags=["bhasha-ui"])


def ui_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the single-page frontend.

    Read from disk per request rather than cached at import time. The file
    is a few kilobytes, this is not a high-traffic service, and editing the
    page during development should not require a server restart.
    """
    if not INDEX.exists():
        raise HTTPException(
            status_code=404,
            detail=f"frontend not found at {INDEX}. The UI is a single static "
                   f"file; see bhasha/app/static/index.html.",
        )
    return HTMLResponse(INDEX.read_text(encoding="utf-8"))


def mount_ui(app, force: bool = False) -> bool:
    """Attach the UI route if it is enabled. Returns whether it was mounted.

    ``force=True`` bypasses the environment flag, for tests and for callers
    that have made their own decision.
    """
    if not (force or ui_enabled()):
        return False
    app.include_router(router)
    return True
