"""A small vector store kept on disk as a NumPy matrix plus JSON metadata.

Fine for thousands of chunks on one machine. Swap for a vector database
(Qdrant, pgvector, Chroma) when the corpus grows.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunker import split_text
from .config import settings
from .embeddings import embedder_id, get_embedder
from .loaders import Section
from .logs import short

log = logging.getLogger("rag.store")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    page: int | None
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, data_dir: Path | None = None):
        self.dir = Path(data_dir or settings.data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.documents: dict[str, dict] = {}
        self.vectors: np.ndarray | None = None
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text())
        if meta.get("embedder") != embedder_id():
            raise RuntimeError(
                f"Index in {self.dir} was built with {meta.get('embedder')}, but the current embedder is "
                f"{embedder_id()}. Delete that folder and re-ingest, or set EMBED_BACKEND/EMBED_MODEL back."
            )
        self.documents = meta["documents"]
        self.chunks = [Chunk(**c) for c in json.loads((self.dir / "chunks.json").read_text())]
        vec_path = self.dir / "vectors.npy"
        self.vectors = np.load(vec_path) if vec_path.exists() and self.chunks else None
        log.info("Loaded index from %s: %d documents, %d chunks", self.dir, len(self.documents), len(self.chunks))

    def _save(self) -> None:
        (self.dir / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False))
        if self.vectors is not None:
            np.save(self.dir / "vectors.npy", self.vectors)
        elif (self.dir / "vectors.npy").exists():
            (self.dir / "vectors.npy").unlink()
        (self.dir / "meta.json").write_text(
            json.dumps({"embedder": embedder_id(), "documents": self.documents}, ensure_ascii=False, indent=1)
        )

    # ---------- writes ----------
    def add_document(self, name: str, sections: list[Section], source: str | None = None) -> dict:
        """Chunk, embed and index a document. Re-adding a document with the same name replaces it."""
        new_chunks: list[Chunk] = []
        doc_id = uuid.uuid4().hex[:12]
        for section in sections:
            for text in split_text(section.text, settings.chunk_size, settings.chunk_overlap):
                new_chunks.append(Chunk(uuid.uuid4().hex[:12], doc_id, name, section.page, text))
        if not new_chunks:
            raise ValueError(f"No text could be extracted from {name}.")

        t0 = time.perf_counter()
        vectors = get_embedder().embed_documents([c.text for c in new_chunks])
        log.debug("Embedded %d chunks of %s in %.2fs", len(new_chunks), name, time.perf_counter() - t0)
        with self._lock:
            for existing_id, doc in list(self.documents.items()):
                if doc["name"] == name:
                    self._remove_locked(existing_id)
            self.chunks.extend(new_chunks)
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
            info = {"id": doc_id, "name": name, "source": source or name, "chunks": len(new_chunks), "added_at": time.time()}
            self.documents[doc_id] = info
            self._save()
        log.info("Indexed %s: %d chunks (avg %d chars)", name, len(new_chunks), sum(len(c.text) for c in new_chunks) // len(new_chunks))
        return info

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            name = self.documents[doc_id]["name"]
            self._remove_locked(doc_id)
            self._save()
            log.info("Removed %s", name)
            return True

    def _remove_locked(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        self.documents.pop(doc_id, None)

    # ---------- reads ----------
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        with self._lock:
            if self.vectors is None or not self.chunks:
                return []
            q = get_embedder().embed_query(query)
            scores = self.vectors @ q  # vectors are normalized, so this is cosine similarity
            top = np.argsort(-scores)[:k]
            hits = [Hit(self.chunks[i], float(scores[i])) for i in top]
        log.debug("Search %r over %d chunks:", short(query, 80), len(self.chunks))
        for rank, h in enumerate(hits, start=1):
            page = f" p.{h.chunk.page}" if h.chunk.page else ""
            log.debug("  #%d score=%.3f %s%s | %s", rank, h.score, h.chunk.doc_name, page, short(h.chunk.text, 90))
        return hits

    def list_documents(self) -> list[dict]:
        return sorted(self.documents.values(), key=lambda d: d["added_at"])
