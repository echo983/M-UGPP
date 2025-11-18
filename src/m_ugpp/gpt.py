"""Lightweight GPT client helpers for UGPP roles."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion
import time

DEFAULT_REQUEST_TIMEOUT = 60.0


class GPTError(RuntimeError):
    """Raised when a GPT request fails or returns invalid data."""


@dataclass
class GPTClient:
    """Thin wrapper around OpenAI Chat Completions with JSON helpers."""

    model: str
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    timeout: float = DEFAULT_REQUEST_TIMEOUT
    prompt_cache_retention: Optional[str] = None  # set to "24h" when model supports it

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise GPTError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        prompt_cache_key: Optional[str] = None,
    ) -> ChatCompletion:
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "response_format": response_format,
            "timeout": self.timeout,
        }
        effective_temp = self.temperature if temperature is None else temperature
        if effective_temp is not None:
            request_kwargs["temperature"] = effective_temp
        if self.prompt_cache_retention:
            request_kwargs["prompt_cache_retention"] = self.prompt_cache_retention
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = prompt_cache_key
        attempts = 3
        delay = 2.0
        for attempt in range(1, attempts + 1):
            try:
                return self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:  # pragma: no cover - delegated to network lib
                if "Connection error" in str(exc) and attempt < attempts:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GPTError(str(exc)) from exc

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: Optional[float] = None,
        prompt_cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        completion = self.chat(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
        )
        content = completion.choices[0].message.content
        if not content:
            raise GPTError("Empty completion content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise GPTError(f"Invalid JSON response: {exc}") from exc
