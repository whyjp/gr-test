"""OpenAI-backed LLMAdapter.

Implements `htb.llm.interfaces.LLMAdapter` by wrapping the official
``openai`` Python client. Requires the ``[llm]`` extra:

    uv sync --extra llm

Env vars honoured (precedence top to bottom):
- ``OPENAI_API_KEY``  (hard requirement)
- ``OPENAI_BASE_URL`` (optional)
- ``EXTRACT_MODEL`` / ``ANSWER_MODEL`` / ``JUDGE_MODEL`` (optional — caller can
  still pass ``model=`` to ``complete()`` to override per-call).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpenAIAdapter:
    """Concrete LLMAdapter using the OpenAI SDK.

    Lazy client construction: the OpenAI client is only built on first
    ``complete()`` call, so unit tests can import this module without the
    SDK being installed or ``OPENAI_API_KEY`` being set.
    """

    default_model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    system_prompt: str | None = None
    max_retries: int = 3
    timeout_s: float = 60.0

    _client: Any = field(default=None, init=False, repr=False)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "openai SDK not installed. Run `uv sync --extra llm` or "
                "`pip install openai`."
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Populate .env or export it."
            )
        base_url = os.environ.get(self.base_url_env) or None
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        client = self._ensure_client()
        effective_model = model or self.default_model
        effective_system = system if system is not None else self.system_prompt

        messages: list[dict[str, str]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def make_default_extract_adapter() -> OpenAIAdapter:
    """Convenience factory reading EXTRACT_MODEL from env (default gpt-4o-mini)."""
    return OpenAIAdapter(default_model=os.environ.get("EXTRACT_MODEL", "gpt-4o-mini"))


def make_default_judge_adapter() -> OpenAIAdapter:
    """Convenience factory reading JUDGE_MODEL from env (default gpt-4o)."""
    return OpenAIAdapter(default_model=os.environ.get("JUDGE_MODEL", "gpt-4o"))
