"""Logging setup. Set LOG_LEVEL=DEBUG in .env to see retrieval scores, chosen passages and prompts."""

from __future__ import annotations

import logging
import os

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party libraries quiet unless they have something important to say.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "asyncio", "huggingface_hub", "anthropic", "fastembed", "multipart", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
