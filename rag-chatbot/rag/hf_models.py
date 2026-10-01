"""List Hugging Face chat models that actually answer with your HF_TOKEN.

    python -m rag.hf_models

Sends a tiny test request to each candidate, so it uses a small amount of your free credits.
"""

from __future__ import annotations

from huggingface_hub import InferenceClient

from .config import settings
from .llm import HF_FALLBACK_MODELS, trending_chat_models


def main() -> None:
    if not settings.hf_token:
        print("Set HF_TOKEN in .env first.")
        return
    client = InferenceClient(provider=settings.hf_provider, token=settings.hf_token, timeout=60)
    candidates = list(dict.fromkeys(HF_FALLBACK_MODELS + trending_chat_models(settings.hf_token)))
    print(f"Testing {len(candidates)} models...\n")
    working = []
    for model in candidates:
        try:
            client.chat_completion(messages=[{"role": "user", "content": "Say OK"}], model=model, max_tokens=5)
            working.append(model)
            print(f"  works   {model}")
        except Exception as e:
            reason = str(e).splitlines()[-1][:90]
            print(f"  no      {model}  ({reason})")
    if working:
        print(f"\nPut this in .env:\nHF_MODEL={working[0]}")
    else:
        print("\nNo model worked. Check your enabled providers at https://huggingface.co/settings/inference-providers")


if __name__ == "__main__":
    main()
