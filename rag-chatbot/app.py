"""Web server: chat UI, document upload, and a streaming chat API.

Run:  uvicorn app:app --reload   then open http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chat import ChatEngine
from rag.config import ROOT, settings
from rag.loaders import load_bytes, load_url
from rag.logs import setup_logging
from rag.store import VectorStore

setup_logging()
log = logging.getLogger("rag.app")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(title="RAG Chatbot")
store = VectorStore()
engine = ChatEngine(store)
log.info(
    "Ready: provider=%s model=%s embedder=%s, %d documents, HF_TOKEN %s, ANTHROPIC_API_KEY %s",
    settings.llm_provider, engine.llm.name, settings.embed_backend if settings.embed_backend == "hash" else settings.embed_model,
    len(store.documents), "set" if settings.hf_token else "not set", "set" if os.getenv("ANTHROPIC_API_KEY") else "not set",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = []


class UrlRequest(BaseModel):
    url: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/documents")
def list_documents() -> dict:
    return {"documents": store.list_documents(), "model": engine.llm.name}


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    added, errors = [], []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append({"name": f.filename, "error": "File is larger than 25 MB."})
            continue
        try:
            added.append(store.add_document(f.filename or "upload", load_bytes(f.filename or "", data)))
        except Exception as e:  # report per file so one bad file doesn't fail the batch
            log.warning("Couldn't index %s: %s", f.filename, e, exc_info=not isinstance(e, ValueError))
            errors.append({"name": f.filename, "error": str(e)})
    return {"added": added, "errors": errors}


@app.post("/api/documents/url")
def add_url(req: UrlRequest) -> dict:
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")
    try:
        title, sections = load_url(req.url)
        return {"added": [store.add_document(title, sections, source=req.url)], "errors": []}
    except Exception as e:
        log.warning("Couldn't index %s: %s", req.url, e, exc_info=not isinstance(e, ValueError))
        raise HTTPException(400, f"Couldn't index {req.url}: {e}")


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not store.remove_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in req.history]

    def events():
        for event in engine.answer_stream(req.question, history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
