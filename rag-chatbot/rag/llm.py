"""Answer-writing models. Each provider streams text and raises LLMError with a readable message."""

from __future__ import annotations

from collections.abc import Iterator

from .config import settings


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


class HuggingFaceLLM:
    """An open model through Hugging Face Inference Providers. Needs HF_TOKEN."""

    def __init__(self, client=None):
        self._client = client

    @property
    def name(self) -> str:
        return settings.hf_model

    @property
    def client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(provider=settings.hf_provider, token=settings.hf_token or None, timeout=120)
        return self._client

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        if not settings.hf_token:
            raise LLMError("HF_TOKEN is missing. Create a token at huggingface.co/settings/tokens and put it in .env.")
        try:
            chunks = self.client.chat_completion(
                messages=[{"role": "system", "content": system}, *messages],
                model=settings.hf_model,
                max_tokens=1500,
                temperature=0.2,
                stream=True,
            )
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise LLMError("Hugging Face rejected HF_TOKEN. Check the token and that it can call Inference Providers.")
            if status == 402:
                raise LLMError("Your Hugging Face free inference credits are used up for this month.")
            if status == 404 or status == 400:
                raise LLMError(f"Hugging Face can't run {settings.hf_model} right now. Try another HF_MODEL in .env. ({e})")
            if status == 429:
                raise LLMError("Rate limited by Hugging Face. Wait a moment and try again.")
            raise LLMError(f"Hugging Face error: {e}")
        except (httpx.HTTPError, OSError) as e:
            raise LLMError(f"Couldn't reach Hugging Face: {e}")


def get_llm():
    return HuggingFaceLLM() if settings.llm_provider == "huggingface" else ClaudeLLM()
