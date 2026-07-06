"""LLM provider adapters.

One interface, three backends. The pipeline only ever sees
LLMProvider.complete_json(prompt) -> dict, so swapping providers is a
config change, not a code change.

Provider selection (env vars):
    ETD_PROVIDER = anthropic | openai | ollama | none   (default: none)
    ANTHROPIC_API_KEY / OPENAI_API_KEY as usual
    ETD_MODEL to override the default model per provider
    OLLAMA_HOST for a non-default Ollama endpoint

"none" disables remote calls entirely; the app then runs on the built-in
heuristic engine, which is also the local-first privacy mode.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

import httpx


class ProviderError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ProviderError(f"Model did not return valid JSON: {text[:200]}")


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 2000) -> str: ...

    def complete_json(self, prompt: str, max_tokens: int = 2000) -> dict:
        return _extract_json(self.complete(prompt, max_tokens))


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ETD_MODEL", "claude-sonnet-4-6")

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.model = model or os.environ.get("ETD_MODEL", "gpt-4o-mini")

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str | None = None):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("ETD_MODEL", "llama3.1")

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = httpx.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def get_provider() -> LLMProvider | None:
    """Return the configured provider, or None for local-only mode."""
    choice = os.environ.get("ETD_PROVIDER", "none").lower()
    if choice in ("", "none", "local"):
        return None
    registry = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }
    if choice not in registry:
        raise ProviderError(f"Unknown ETD_PROVIDER: {choice}")
    return registry[choice]()
