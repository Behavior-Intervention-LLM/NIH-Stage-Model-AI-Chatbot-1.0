"""
LLM client wrapper.
Supports vLLM(OpenAI-compatible) backend.
"""
import json
import re
from typing import Any, Dict, Optional

import requests

from app.config import settings


class LLMClient:
    """Unified LLM client that can also parse JSON responses."""

    def __init__(self):
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.vllm_base_url = settings.VLLM_BASE_URL.rstrip("/")
        self.api_key = settings.LLM_API_KEY

    def is_enabled(self) -> bool:
        return True

    def chat_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.is_enabled():
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.vllm_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def chat_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Parse model text output into JSON when possible.
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

