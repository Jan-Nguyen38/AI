"""Web server: chat UI, document upload, and a streaming chat API.

Run:  uvicorn app:app --reload   then open http://localhost:8000
"""

from __future__ import annotations

import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chat import ChatEngine
from rag.config import ROOT, settings
from rag.loaders import load_bytes, load_url
from rag.store import VectorStore

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(title="RAG Chatbot")
store = VectorStore()
engine = ChatEngine(store)


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
    return {"documents": store.list_documents(), "model": settings.claude_model}


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
