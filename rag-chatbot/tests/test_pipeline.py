"""Offline tests: keyword embeddings and a fake Claude client, so no network or API key is needed."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["EMBED_BACKEND"] = "hash"
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from rag.chat import IDK, ChatEngine, cited_ids  # noqa: E402
from rag.chunker import split_text  # noqa: E402
from rag.llm import ClaudeLLM, HuggingFaceLLM  # noqa: E402
from rag.loaders import load_path  # noqa: E402
from rag.store import VectorStore  # noqa: E402

SAMPLES = ROOT / "sample_docs"


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield from (self.text[i : i + 7] for i in range(0, len(self.text), 7))

    def get_final_message(self):
        return SimpleNamespace(stop_reason="end_turn")


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.reply)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path)
    for f in sorted(SAMPLES.iterdir()):
        s.add_document(f.name, load_path(f), source=str(f))
    return s


def test_chunker_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 360 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]


def test_docx_tables_are_extracted():
    text = load_path(SAMPLES / "security_policy.docx")[0].text
    assert "Hardware security module only" in text


def test_search_finds_the_right_document(store):
    hit = store.search("How long does the robot battery last on a charge?")[0]
    assert hit.chunk.doc_name == "product_faq.txt"


def test_index_persists_and_replaces_same_name(store, tmp_path):
    reopened = VectorStore(tmp_path)
    assert len(reopened.list_documents()) == 3
    reopened.add_document("product_faq.txt", load_path(SAMPLES / "product_faq.txt"))
    assert len(reopened.list_documents()) == 3


def test_answer_streams_text_and_returns_cited_sources(store):
    client = FakeClient("A full charge lasts about 10 hours [1].")
    events = list(ChatEngine(store, ClaudeLLM(client)).answer_stream("How long does the battery last?"))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    done = events[-1]
    assert text == "A full charge lasts about 10 hours [1]."
    assert done["type"] == "done" and done["grounded"]
    assert [s["id"] for s in done["sources"]] == [1]
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "<sources>" in prompt and "Question: How long does the battery last?" in prompt


def test_unrelated_question_says_idk_without_calling_claude(store):
    client = FakeClient("should not be used")
    for q in ["Quelle est la recette du pot-au-feu ?", "What's the best pizza in Naples?"]:
        events = list(ChatEngine(store, ClaudeLLM(client)).answer_stream(q))
        assert events[0]["text"] == IDK, q
    assert events[-1] == {"type": "done", "sources": [], "grounded": False}
    assert client.calls == []


def test_follow_up_uses_history(store):
    client = FakeClient("It costs 1,900 USD per month [1].")
    history = [
        {"role": "user", "content": "Tell me about the Atlas R2 warehouse robot subscription"},
        {"role": "assistant", "content": "It is an autonomous mobile robot [1]."},
    ]
    events = list(ChatEngine(store, ClaudeLLM(client)).answer_stream("How much is it per month?", history))
    assert events[-1]["type"] == "done"
    msgs = client.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "product_faq.txt" in msgs[-1]["content"]


class FakeHFClient:
    def __init__(self, reply, unsupported=()):
        self.reply = reply
        self.unsupported = set(unsupported)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.unsupported:
            import httpx
            from huggingface_hub.errors import HfHubHTTPError

            response = httpx.Response(400, request=httpx.Request("POST", "https://router.huggingface.co"))
            raise HfHubHTTPError("Bad request: model_not_supported by any provider you have enabled", response=response)
        return self._chunks()

    def _chunks(self):
        for i in range(0, len(self.reply), 5):
            delta = SimpleNamespace(content=self.reply[i : i + 5])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_hugging_face_provider_streams_with_system_prompt(store, monkeypatch):
    from dataclasses import replace

    from rag import llm

    monkeypatch.setattr(llm, "settings", replace(llm.settings, hf_token="hf_test", hf_model="org/model"))
    client = FakeHFClient("Up to 5 days can be carried over [1].")
    events = list(ChatEngine(store, HuggingFaceLLM(client)).answer_stream("How many vacation days carry over?"))
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Up to 5 days can be carried over [1]."
    assert events[-1]["sources"][0]["document"] == "employee_handbook.md"
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"


def test_hugging_face_auto_skips_unsupported_models(store, monkeypatch):
    from dataclasses import replace

    from rag import llm

    monkeypatch.setattr(llm, "settings", replace(llm.settings, hf_token="hf_test", hf_model="auto"))
    monkeypatch.setattr(llm, "trending_chat_models", lambda token: [])
    first, second = llm.HF_FALLBACK_MODELS[:2]
    client = FakeHFClient("Up to 5 days [1].", unsupported={first})
    engine = ChatEngine(store, HuggingFaceLLM(client))
    events = list(engine.answer_stream("How many vacation days carry over?"))
    assert events[-1]["type"] == "done"
    assert [c["model"] for c in client.calls] == [first, second]
    assert engine.llm.name == second
    list(engine.answer_stream("How many vacation days carry over?"))
    assert client.calls[-1]["model"] == second  # remembers the model that worked


def test_hugging_face_fixed_model_unsupported_explains_fix(store, monkeypatch):
    from dataclasses import replace

    from rag import llm

    monkeypatch.setattr(llm, "settings", replace(llm.settings, hf_token="hf_test", hf_model="org/missing"))
    client = FakeHFClient("x", unsupported={"org/missing"})
    events = list(ChatEngine(store, HuggingFaceLLM(client)).answer_stream("How many vacation days carry over?"))
    assert events[-1]["type"] == "error" and "HF_MODEL=auto" in events[-1]["message"]


def test_hugging_face_without_token_reports_error(store, monkeypatch):
    from dataclasses import replace

    from rag import llm

    monkeypatch.setattr(llm, "settings", replace(llm.settings, hf_token=""))
    events = list(ChatEngine(store, HuggingFaceLLM(FakeHFClient("x"))).answer_stream("How many vacation days carry over?"))
    assert events[-1]["type"] == "error" and "HF_TOKEN" in events[-1]["message"]


def test_cited_ids():
    assert cited_ids("a [2] b [1][2] c [10]") == [1, 2, 10]


def test_api_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module

    s = VectorStore(tmp_path)
    monkeypatch.setattr(app_module, "store", s)
    monkeypatch.setattr(app_module, "engine", ChatEngine(s, ClaudeLLM(FakeClient("Core hours are 10:00 to 15:00 [1]."))))
    c = TestClient(app_module.app)

    with open(SAMPLES / "employee_handbook.md", "rb") as f:
        r = c.post("/api/documents", files=[("files", ("employee_handbook.md", f, "text/markdown"))])
    assert r.status_code == 200 and len(r.json()["added"]) == 1

    r = c.post("/api/documents", files=[("files", ("bad.exe", b"x", "application/octet-stream"))])
    assert r.json()["errors"]

    r = c.post("/api/chat", json={"question": "What are the core working hours?", "history": []})
    assert "Core hours" in r.text and '"type": "done"' in r.text

    doc_id = c.get("/api/documents").json()["documents"][0]["id"]
    assert c.delete(f"/api/documents/{doc_id}").status_code == 200
    assert c.get("/api/documents").json()["documents"] == []
    assert c.get("/").status_code == 200
