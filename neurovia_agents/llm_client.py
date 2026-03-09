"""
NeuroVia LLM Client — Unified interface to multiple LLM providers.
Priority: Groq → Gemini → Grok → Ollama (local fallback).
"""

import os
import json
import logging
import httpx
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("neurovia.llm")


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class LLMClient:
    """
    Unified LLM client with automatic failover across providers.
    
    Priority order:
      1. Groq  (Llama 3.3 70B — fast inference)
      2. Gemini (Google — good reasoning)
      3. Ollama (local fallback — no API key needed)
    """

    PROVIDERS = [
        {
            "name": "groq",
            "env_key": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "llama-3.3-70b-versatile",
            "headers_fn": lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        },
        {
            "name": "gemini",
            "env_key": "GEMINI_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "model": "gemini-2.0-flash",
            "headers_fn": lambda key: {"Content-Type": "application/json"},
        },
        {
            "name": "ollama",
            "env_key": None,  # No key needed
            "base_url": None,  # Set from env
            "model": "llama3.1",
            "headers_fn": lambda key: {"Content-Type": "application/json"},
        },
    ]

    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._available_providers = self._detect_providers()
        log.info(f"Available LLM providers: {[p['name'] for p in self._available_providers]}")

    def _detect_providers(self) -> list[dict]:
        """Detect which providers have valid configuration."""
        available = []
        for p in self.PROVIDERS:
            if p["name"] == "ollama":
                base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                p = {**p, "base_url": f"{base}/api/chat"}
                available.append(p)
            elif p["env_key"] and os.environ.get(p["env_key"]):
                available.append(p)
        return available

    def complete(self, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int = 4096, json_mode: bool = False) -> LLMResponse:
        """
        Send a chat completion request, trying providers in priority order.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            json_mode: If True, instruct the model to return valid JSON
        
        Returns:
            LLMResponse with the completion content
        
        Raises:
            RuntimeError: If all providers fail
        """
        errors = []

        for provider in self._available_providers:
            try:
                if provider["name"] == "groq":
                    return self._call_groq(provider, messages, temperature, max_tokens, json_mode)
                elif provider["name"] == "gemini":
                    return self._call_gemini(provider, messages, temperature, max_tokens, json_mode)
                elif provider["name"] == "ollama":
                    return self._call_ollama(provider, messages, temperature, max_tokens, json_mode)
            except Exception as e:
                log.warning(f"Provider {provider['name']} failed: {e}")
                errors.append(f"{provider['name']}: {e}")
                continue

        raise RuntimeError(
            f"All LLM providers failed:\n" + "\n".join(errors)
        )

    def _call_groq(self, provider: dict, messages: list, temperature: float,
                   max_tokens: int, json_mode: bool) -> LLMResponse:
        key = os.environ.get(provider["env_key"], "")
        headers = provider["headers_fn"](key)

        body = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = self._client.post(provider["base_url"], headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider="groq",
            model=provider["model"],
            usage=data.get("usage", {}),
            raw=data,
        )

    def _call_gemini(self, provider: dict, messages: list, temperature: float,
                     max_tokens: int, json_mode: bool) -> LLMResponse:
        key = os.environ.get(provider["env_key"], "")
        url = provider["base_url"].format(model=provider["model"]) + f"?key={key}"
        headers = provider["headers_fn"](key)

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        resp = self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResponse(
            content=content,
            provider="gemini",
            model=provider["model"],
            usage=data.get("usageMetadata", {}),
            raw=data,
        )

    def _call_ollama(self, provider: dict, messages: list, temperature: float,
                     max_tokens: int, json_mode: bool) -> LLMResponse:
        body = {
            "model": provider["model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"

        resp = self._client.post(provider["base_url"], json=body)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data["message"]["content"],
            provider="ollama",
            model=provider["model"],
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )

    def close(self):
        self._client.close()

    @property
    def available_providers(self) -> list[str]:
        return [p["name"] for p in self._available_providers]


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
