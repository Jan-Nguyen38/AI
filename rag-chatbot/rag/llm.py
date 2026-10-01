"""Answer-writing models. Each provider streams text and raises LLMError with a readable message."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from .config import settings


log = logging.getLogger("rag.llm")


class LLMError(Exception):
    pass


class ClaudeLLM:
    """Claude through the Anthropic API. Needs ANTHROPIC_API_KEY."""

    def __init__(self, client=None):
        self._client = client

    @property
    def name(self) -> str:
        return settings.claude_model

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        import anthropic

        try:
            with self.client.beta.messages.stream(
                model=settings.claude_model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={"effort": settings.claude_effort},
                # Retry on another model if the main one declines for a safety reason.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.AuthenticationError:
            raise LLMError("Claude API key is missing or invalid. Set ANTHROPIC_API_KEY in .env.")
        except anthropic.RateLimitError:
            raise LLMError("Rate limited by the Claude API. Wait a moment and try again.")
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise LLMError("Couldn't reach the Claude API. Check your network connection.")
        if final.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this question.")


# Tried in order when HF_MODEL=auto, before trending models from the Hub.
# Which ones work depends on the inference providers enabled on your account.
HF_FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


def trending_chat_models(token: str | None, limit: int = 15) -> list[str]:
    """Popular text-generation models that at least one inference provider serves."""
    try:
        from huggingface_hub import list_models

        models = list_models(
            inference_provider="all", pipeline_tag="text-generation", sort="trending_score", limit=limit, token=token
        )
        return [m.id for m in models]
    except Exception:
        return []


def _is_model_unavailable(status: int | None, error: Exception) -> bool:
    text = str(error).lower()
    return status in (400, 404) and ("model" in text or "not supported" in text or "not found" in text)


class HuggingFaceLLM:
    """An open model through Hugging Face Inference Providers. Needs HF_TOKEN."""

    def __init__(self, client=None):
        self._client = client
        self._working_model: str | None = None

    @property
    def name(self) -> str:
        return self._working_model or settings.hf_model

    @property
    def client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(provider=settings.hf_provider, token=settings.hf_token or None, timeout=120)
        return self._client

    def _candidates(self) -> list[str]:
        if self._working_model:
            return [self._working_model]
        if settings.hf_model.lower() != "auto":
            return [settings.hf_model]
        seen, out = set(), []
        for m in HF_FALLBACK_MODELS + trending_chat_models(settings.hf_token or None):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        if not settings.hf_token:
            raise LLMError("HF_TOKEN is missing. Create a token at huggingface.co/settings/tokens and put it in .env.")
        tried = []
        for model in self._candidates():
            log.debug("Trying Hugging Face model %s (provider=%s)", model, settings.hf_provider)
            try:
                chunks = self.client.chat_completion(
                    messages=[{"role": "system", "content": system}, *messages],
                    model=model,
                    max_tokens=1500,
                    temperature=0.2,
                    stream=True,
                )
                started = False
                for chunk in chunks:
                    if not started:
                        started = True
                        if self._working_model != model:
                            log.info("Using Hugging Face model %s", model)
                        self._working_model = model
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except HfHubHTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if _is_model_unavailable(status, e):
                    log.warning("Hugging Face model %s not available (HTTP %s), trying the next one", model, status)
                    # Nothing was streamed yet, so it's safe to move on to the next model.
                    tried.append(model)
                    continue
                if status == 401:
                    raise LLMError("Hugging Face rejected HF_TOKEN. Check the token and that it can call Inference Providers.")
                if status == 402:
                    raise LLMError("Your Hugging Face free inference credits are used up for this month.")
                if status == 429:
                    raise LLMError("Rate limited by Hugging Face. Wait a moment and try again.")
                log.error("Hugging Face HTTP %s with %s: %s", status, model, e)
                raise LLMError(f"Hugging Face error with {model}: {e}")
            except (httpx.HTTPError, OSError) as e:
                raise LLMError(f"Couldn't reach Hugging Face: {e}")
        raise LLMError(
            f"None of the enabled Hugging Face providers serve {', '.join(tried)}. "
            "Set HF_MODEL=auto in .env, or run `python -m rag.hf_models` to list models that work with your token."
        )


def get_llm():
    return HuggingFaceLLM() if settings.llm_provider == "huggingface" else ClaudeLLM()
