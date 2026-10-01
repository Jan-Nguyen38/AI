"""Split text into overlapping chunks along paragraph and sentence boundaries."""

from __future__ import annotations

import re

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    pieces = _split_recursive(text, size, 0)
    return _merge(pieces, size, overlap)


def _split_recursive(text: str, size: int, level: int) -> list[str]:
    if len(text) <= size:
        return [text]
    if level >= len(_SEPARATORS):
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep = _SEPARATORS[level]
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator attached so sentences stay readable.
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > size:
            out.extend(_split_recursive(piece, size, level + 1))
        elif piece:
            out.append(piece)
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            # Start the next chunk with the tail of the previous one for context.
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
