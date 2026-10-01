"""Local text embeddings.

The default backend runs a small embedding model on your CPU with fastembed (ONNX),
so no extra account is needed. The model (~130 MB) downloads from Hugging Face on
first use. The "hash" backend is a keyword-only fallback for offline testing.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from .config import settings


class Embedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class FastEmbedder(Embedder):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed(text)), dtype=np.float32))[0]


_STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or our s so that the "
    "their there this to was we what when where which who why will with you your about best what's whats".split()
)


class HashEmbedder(Embedder):
    """Bag-of-words hashed into a fixed vector. Matches keywords, not meaning."""

    dim = 2048

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            if len(tok) < 2 or tok in _STOPWORDS:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embed_backend == "hash":
        return HashEmbedder()
    return FastEmbedder(settings.embed_model)


def embedder_id() -> str:
    return "hash" if settings.embed_backend == "hash" else f"fastembed:{settings.embed_model}"
