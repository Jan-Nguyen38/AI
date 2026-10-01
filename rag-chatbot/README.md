# RAG Chatbot (v1)

Chat with your own documents. Upload PDFs, Word files, text or web pages, then ask questions. Answers are written by an open model on Hugging Face or by Claude, come only from your documents, cite the passages they use, and say "I don't know" when the documents don't cover the question.

## What's in this version

- Upload PDF, `.docx`, `.txt`, `.md`, `.html` files, or add a web page by URL
- Documents are split into overlapping chunks and indexed with a local embedding model (no extra account needed)
- Search by meaning, not just keywords
- Answers from a Hugging Face model or Claude, grounded in the retrieved passages, with clickable `[1]` citations that open the source text
- Chat history, so follow-up questions keep context (kept in your browser)
- "I don't know" when nothing relevant is found (no model call is made) or when the passages don't answer the question
- Streaming answers, and a document list where you can remove documents

## Pick the model that writes answers

| Option | What you need | Cost |
|---|---|---|
| **Hugging Face** (open models, picked automatically) | A free account at https://huggingface.co and an access token | Free monthly credits, then pay as you go |
| **Claude** (`claude-opus-5-5`) | An API key from https://platform.claude.com. A Claude.ai subscription doesn't include API access | Pay per use |

To get a Hugging Face token: sign in, go to https://huggingface.co/settings/tokens, click **Create new token**, choose **Fine-grained**, tick **Make calls to Inference Providers**, create it and copy the value (it starts with `hf_`).

## Run it on Windows (PowerShell)

You need Python 3.10 or newer (`python --version`). In VS Code, open the repository folder and a PowerShell terminal, then:

```powershell
cd rag-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

In `.env`, paste your token after `HF_TOKEN=` (for Claude, use option B instead) and save. Then:

```powershell
python -m rag.ingest sample_docs
uvicorn app:app --reload
```

Open http://localhost:8000.

### Which Hugging Face models work for me?

Which models you can use depends on the inference providers enabled on your account (https://huggingface.co/settings/inference-providers). To see which ones answer with your token:

```powershell
python -m rag.hf_models
```

It tests each candidate with a tiny request and prints the `HF_MODEL=` line to put in `.env`. You can also browse https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation&sort=trending

If `Activate.ps1` is blocked with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

## Run it on macOS or Linux

```bash
cd rag-chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in HF_TOKEN or ANTHROPIC_API_KEY
python -m rag.ingest sample_docs/
uvicorn app:app --reload
```

The first ingest downloads the embedding model (about 130 MB) from Hugging Face; later runs use the cached copy.

Keep tokens and keys in `.env` (ignored by git) or your environment, never in code.

### Try these with the sample documents

The `sample_docs/` folder holds three fictional documents for a made-up company, Lumen Robotics: an employee handbook, a product FAQ and a security policy (Word, with a table).

- How long does the Atlas R2 battery last? Then: and how long does charging take?
- How many vacation days can I carry over?
- Can I paste customer contracts into an AI tool?
- What's the CEO's salary? (should say it doesn't know)

## Command line

```bash
python -m rag.ingest path/to/folder        # every supported file, recursively
python -m rag.ingest report.pdf notes.docx
python -m rag.ingest https://example.com/faq
python -m rag.ingest --list
```

Adding a document with the same name again replaces the old version.

## Settings

Set these in `.env` or the environment.

| Variable | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `huggingface` if only `HF_TOKEN` is set, else `anthropic` | Which model writes answers |
| `HF_TOKEN` | | Hugging Face access token (for `huggingface`) |
| `HF_MODEL` | `auto` | `auto` tries well-known chat models (`openai/gpt-oss-120b` first) and then trending ones until one is served by your enabled providers, and remembers it. Or name any chat model on Hugging Face Inference Providers |
| `HF_PROVIDER` | `auto` | Which inference provider runs it; `auto` picks one for you |
| `ANTHROPIC_API_KEY` | | Claude API key (for `anthropic`) |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model that writes answers |
| `CLAUDE_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh` or `max`. Raise it for harder documents; it costs more and is slower |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any [fastembed model](https://qdrant.github.io/fastembed/examples/Supported_Models/). For Vietnamese or other languages try `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `EMBED_BACKEND` | `fastembed` | `hash` is a keyword-only fallback that works offline, for testing |
| `TOP_K` | `5` | Passages sent to the model per question |
| `MIN_SCORE` | `0.45` | Passages below this similarity are ignored. If none pass, the bot says it doesn't know without calling the model. Lower it if good questions get "I don't know" |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Chunk length and overlap, in characters |
| `DATA_DIR` | `./data` | Where the index is stored |

Changing `EMBED_MODEL` or `EMBED_BACKEND` needs a fresh index: delete `data/` and ingest again.

## How it works

```
upload ─► loaders.py (text + PDF page numbers) ─► chunker.py (overlapping chunks)
       ─► embeddings.py (local vectors) ─► store.py (NumPy matrix + JSON on disk)

question ─► store.search (cosine similarity, top K, minimum score)
         ─► chat.py: numbered <source> blocks + chat history ─► llm.py: Hugging Face or Claude (streaming)
         ─► answer with [n] citations ─► UI shows only the cited sources
```

- `rag/chat.py` holds the system prompt. The model is told to use only the sources, cite them, reply with an exact "I don't know" sentence when they don't answer, and treat document text as data, never as instructions.
- Follow-up questions are searched together with the previous question, so "how long does charging take?" still finds the robot FAQ.
- With Claude, requests opt into server-side fallback (`fallbacks: "default"`), so if the main model declines a request for a safety reason, the API retries on another model in the same call.

## API

| Method | Path | Body |
|---|---|---|
| `GET` | `/api/documents` | |
| `POST` | `/api/documents` | multipart `files` |
| `POST` | `/api/documents/url` | `{"url": "..."}` |
| `DELETE` | `/api/documents/{id}` | |
| `POST` | `/api/chat` | `{"question": "...", "history": [{"role": "user", "content": "..."}, ...]}`, returns server-sent events: `text`, then `done` with `sources`, or `error` |

## Tests

```bash
pytest -q
```

The tests run offline with the keyword embedder and fake Claude and Hugging Face clients, so they need no key or network.

## Known limits of v1

- Scanned PDFs and images aren't read (no OCR yet)
- One shared document collection with no login; anyone who can open the page can upload or delete
- The index is a single file set on disk, good for thousands of chunks; move to a vector database for more
- No hybrid keyword search or reranking yet

Those are in the next feature groups (better answers, users and admin, running it).
