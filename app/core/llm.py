"""
Unified LLM client supporting OpenAI, Anthropic, Ollama, and Groq.
"""
import json
import re
from typing import Any, Dict, Iterator, Optional

import requests
from openai import OpenAI

from app.config import settings


class LLMClient:
    """Single client for all supported LLM providers. Handles JSON parsing."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self._openai_client: Optional[OpenAI] = None
        # Models that reject a custom temperature (e.g. reasoning models);
        # learned at runtime from 400 responses, then skipped thereafter.
        self._no_temperature_models: set[str] = set()

    def is_enabled(self) -> bool:
        return self.provider in {"ollama", "anthropic", "openai", "groq"}

    def chat_text(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Optional[str]:
        if not self.is_enabled():
            return None

        if self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, model=model)
        if self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt)

        return None

    def chat_text_stream(
        self, system_prompt: str, user_prompt: str, model: Optional[str] = None
    ) -> Iterator[str]:
        """Yield response text incrementally. Falls back to a single chunk for
        providers without streaming support here."""
        if self.provider == "openai" and self.is_enabled():
            stream = self._openai_completion(system_prompt, user_prompt, model=model, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return

        text = self.chat_text(system_prompt, user_prompt, model=model)
        if text:
            yield text

    def _get_openai_client(self) -> OpenAI:
        """Reuse one client (and its HTTP connection pool) across calls."""
        if self._openai_client is None:
            api_key = settings.OPENAI_API_KEY or settings.LLM_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self._openai_client = OpenAI(api_key=api_key, base_url=settings.OPENAI_BASE_URL or None)
        return self._openai_client

    def _openai_completion(
        self, system_prompt: str, user_prompt: str, model: Optional[str] = None, stream: bool = False
    ):
        from openai import BadRequestError

        client = self._get_openai_client()
        target_model = model or self.model
        kwargs: Dict[str, Any] = {
            "model": target_model,
            "max_completion_tokens": settings.LLM_MAX_TOKENS,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if target_model not in self._no_temperature_models:
            kwargs["temperature"] = settings.LLM_TEMPERATURE
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "temperature" in str(exc) and "temperature" in kwargs:
                self._no_temperature_models.add(target_model)
                kwargs.pop("temperature")
                return client.chat.completions.create(**kwargs)
            raise

    def _call_openai(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Optional[str]:
        response = self._openai_completion(system_prompt, user_prompt, model=model)
        return (response.choices[0].message.content or "").strip()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": settings.LLM_TEMPERATURE},
        }
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        api_key = settings.ANTHROPIC_API_KEY or settings.LLM_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        model = settings.ANTHROPIC_MODEL
        payload = {
            "model": model,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("content", [{}])[0].get("text", "").strip()

    def _call_groq(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        payload = {
            "model": settings.GROQ_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def chat_json(
        self, system_prompt: str, user_prompt: str, model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Call chat_text and parse the result as JSON."""
        raw = self.chat_text(system_prompt=system_prompt, user_prompt=user_prompt, model=model)
        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        brace = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(1))
            except Exception:
                return None

        return None


llm_client = LLMClient()