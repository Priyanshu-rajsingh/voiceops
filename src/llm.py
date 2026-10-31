"""Minimal, provider-agnostic LLM client built on `requests`.

Supports the Anthropic Messages API and any OpenAI-compatible Chat Completions
API (OpenAI, Gemini, Groq, OpenRouter, Ollama, ...). Configure through .env:
LLM_PROVIDER, LLM_MODEL, LLM_API_KEY (or the provider's own key variable).
"""
from __future__ import annotations

import json
import re
import time

import requests

from . import config


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError(f"no JSON object in model reply: {text[:200]!r}")


class LLM:
    MAX_ATTEMPTS = 8

    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None, timeout: int = 90):
        s = config.get_settings()
        self.provider = (provider or s.llm_provider).lower()
        self.model = model or s.llm_model
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.timeout = timeout
        self.calls = 0
        if not self.model:
            raise LLMError("LLM_MODEL is not set (see .env.example).")
        if not self.api_key and self.provider != "ollama":
            raise LLMError("No LLM API key found. Set LLM_API_KEY (or the provider-specific key) in .env.")

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, json_mode: bool = False,
             max_tokens: int = 900, temperature: float = 0.0) -> str:
        self.calls += 1
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens, temperature)
        return self._openai(system, user, json_mode, max_tokens, temperature)

    def chat_json(self, system: str, user: str, **kw) -> dict:
        last: Exception | None = None
        for _ in range(2):  # one repair attempt if the model returns malformed JSON
            try:
                return parse_json(self.chat(system, user, json_mode=True, **kw))
            except LLMError as e:
                last = e
        raise LLMError(str(last))

    # ------------------------------------------------------------------
    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        wait = 1.5
        last_attempt = self.MAX_ATTEMPTS - 1
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == last_attempt:
                    raise LLMError(f"network error: {e}") from e
                time.sleep(wait)
                wait = min(wait * 2, 30)
                continue
            if r.status_code == 200:
                return r.json()
            body = r.text[:400]
            if r.status_code == 400 and attempt < last_attempt:
                low = body.lower()
                changed = False
                if "temperature" in low and "temperature" in payload:
                    payload.pop("temperature")
                    changed = True
                if "max_completion_tokens" in low and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    changed = True
                if ("response_format" in low or "json_object" in low) and "response_format" in payload:
                    payload.pop("response_format")
                    changed = True
                if "reasoning_effort" in low and "reasoning_effort" in payload:
                    payload.pop("reasoning_effort")
                    changed = True
                if changed:
                    continue
            if r.status_code in (429, 500, 502, 503, 529) and attempt < last_attempt:
                retry = r.headers.get("retry-after")
                try:
                    delay = float(retry) if retry else wait
                except ValueError:
                    delay = wait
                time.sleep(min(delay, 60))
                wait = min(wait * 2, 30)
                continue
            raise LLMError(f"HTTP {r.status_code} from {self.provider}: {body}")
        raise LLMError("retries exhausted")

    def _anthropic(self, system, user, max_tokens, temperature) -> str:
        base = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        data = self._post(
            f"{base}/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
             "system": system, "messages": [{"role": "user", "content": user}]},
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _openai(self, system, user, json_mode, max_tokens, temperature) -> str:
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model.lower():
            payload["reasoning_effort"] = "low"  # keep hidden reasoning short so the answer fits in the limit
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        for _ in range(2):  # a reasoning model can burn the whole limit on thinking: retry once with more room
            data = self._post(url, headers, payload)
            try:
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
            except (KeyError, IndexError, AttributeError) as e:
                raise LLMError(f"unexpected response shape: {str(data)[:200]}") from e
            if text.strip() or choice.get("finish_reason") != "length":
                return text
            key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
            payload[key] = min(int(payload.get(key, 900)) * 3, 6000)
        return text