"""
LLM （ Ollama，）
"""
import json
import re
from typing import Any, Dict, Optional

import requests

from app.config import settings


class LLMClient:
    """ LLM ， JSON """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")

    def is_enabled(self) -> bool:
        return self.provider in {"ollama"}

    def chat_text(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.is_enabled():
            return None

        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                },
            }
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()

        return None

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

