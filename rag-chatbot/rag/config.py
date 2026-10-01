"""Settings, read from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    # Chat Q&A does well at low effort; raise to "medium"/"high" for harder documents.
    claude_effort: str = os.getenv("CLAUDE_EFFORT", "low")
    embed_backend: str = os.getenv("EMBED_BACKEND", "fastembed")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    # Chunks scoring below this are treated as "not relevant". If none pass,
    # the bot says it doesn't know without calling Claude.
    min_score: float = float(
        os.getenv("MIN_SCORE", "0.45" if os.getenv("EMBED_BACKEND", "fastembed") == "fastembed" else "0.08")
    )
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))


settings = Settings()
