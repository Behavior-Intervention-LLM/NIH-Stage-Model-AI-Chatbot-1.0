"""
LLM client supporting Ollama (local), Anthropic, and OpenAI.
"""
import json
import re
from typing import Any, Dict, Optional
from openai import OpenAI

import requests

from app.config import settings


class LLMClient:
    """Unified LLM client. Parses JSON from model output when needed."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    
    def is_enabled(self) -> bool:
        return self.provider in {"ollama", "anthropic", "openai", "groq"}

    def chat_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.is_enabled():
            return None

        if self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)

        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)

        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)

        if self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt)

        return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:

        api_key = settings.OPENAI_API_KEY or settings.LLM_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        else:
            print(f"OAI key set: {api_key[:8]}...")
        
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            temperature=settings.LLM_TEMPERATURE,
            max_completion_tokens=settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()



    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
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
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        api_key = settings.ANTHROPIC_API_KEY or settings.LLM_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        model = settings.ANTHROPIC_MODEL if self.model == "qwen2.5:3b-instruct" else self.model
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
        model = settings.GROQ_MODEL
        payload = {
            "model": model,
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

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        output JSON，。
        """
        raw = self.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)
        if not raw:
            return None

        # 1) 
        try:
            return json.loads(raw)
        except Exception:
            pass

        # 2)  ```json ... ```
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        # 3)  {...}
        brace = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(1))
            except Exception:
                return None

        return None


llm_client = LLMClient()

