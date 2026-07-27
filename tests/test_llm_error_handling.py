"""Regression test: a live run hit `httpx.ReadTimeout` propagating straight
through chat()/generate_json() uncaught, because the original code only
caught the builtin `TimeoutError` -- which `httpx.ReadTimeout` is not a
subclass of. No network involved here: `ollama.Client.chat` is monkeypatched
to raise the exact exception observed live.
"""

from __future__ import annotations

import httpx
import ollama
import pytest
from pydantic import BaseModel

from reflect import llm


class _Schema(BaseModel):
    value: int


def test_chat_wraps_httpx_read_timeout_as_llm_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ollama.Client, "chat", raise_timeout)

    with pytest.raises(llm.LLMError, match="timed out"):
        llm.chat([{"role": "user", "content": "hi"}], temperature=0.0)


def test_generate_json_wraps_httpx_read_timeout_as_llm_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ollama.Client, "chat", raise_timeout)

    with pytest.raises(llm.LLMError, match="timed out"):
        llm.generate_json(prompt="hi", schema=_Schema)


def test_chat_wraps_builtin_timeout_error_as_llm_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("socket timed out")

    monkeypatch.setattr(ollama.Client, "chat", raise_timeout)

    with pytest.raises(llm.LLMError, match="timed out"):
        llm.chat([{"role": "user", "content": "hi"}], temperature=0.0)


def test_chat_wraps_ollama_response_error(monkeypatch):
    def raise_response_error(*args, **kwargs):
        raise ollama.ResponseError("model not found")

    monkeypatch.setattr(ollama.Client, "chat", raise_response_error)

    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}], temperature=0.0)
