"""Thin wrapper around the Ollama client: chat and JSON-mode structured
generation with Pydantic validation + one retry.

All live-LLM calls (generator, judge, critic) go through this module. It is
the single place that has to handle Ollama's failure modes. The Ollama
python client raises `ollama.ResponseError`/`ollama.RequestError` for
HTTP-level failures, but under real load against a server shared with other
builders, a stalled read surfaces as a raw `httpx.TimeoutException`
(specifically `httpx.ReadTimeout`) that ollama's client does NOT catch or
wrap -- confirmed live: a 240s-timeout call during a contended run raised
`httpx.ReadTimeout` straight through this module uncaught, because an
earlier version here only caught the *builtin* `TimeoutError`, which
`httpx.ReadTimeout` is not a subclass of. Both `httpx.TimeoutException` and
the builtin `TimeoutError` (in case a lower layer ever raises the bare
socket version) are caught here and re-raised as a single `LLMError`
callers can handle uniformly.
"""

from __future__ import annotations

import json

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from reflect.config import CONFIG

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised for any failure talking to the local Ollama server."""


def _client() -> ollama.Client:
    return ollama.Client(host=CONFIG.ollama_host, timeout=CONFIG.llm_timeout_s)


def chat(messages: list[Message], temperature: float) -> str:
    """Single chat completion. Returns the assistant's text content."""
    try:
        response = _client().chat(
            model=CONFIG.model,
            messages=messages,
            options={"temperature": temperature},
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise LLMError(f"Ollama chat call failed: {exc}") from exc
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise LLMError(f"Ollama chat call timed out after {CONFIG.llm_timeout_s}s: {exc}") from exc
    return response["message"]["content"]


def generate_json(
    prompt: str,
    schema: type[BaseModel],
    system: str | None = None,
    temperature: float = 0.0,
    max_retries: int = CONFIG.llm_max_retries,
) -> BaseModel:
    """Call the model in JSON mode and validate the result against `schema`,
    retrying (by default once) with the validation error fed back to the
    model if parsing/validation fails.

    Used by judge.py and critic.py so their outputs are structured data from
    the start, never free text parsed downstream -- the lesson from the
    memory-agent project (05) is that LLM output fed back into another LLM
    call compounds errors when it drifts from schema.
    """
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    raw = ""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _client().chat(
                model=CONFIG.model,
                messages=messages,
                format="json",
                options={"temperature": temperature},
            )
            raw = response["message"]["content"]
            data = json.loads(raw)
            return schema.model_validate(data)
        except (ollama.RequestError, ollama.ResponseError) as exc:
            raise LLMError(f"Ollama JSON call failed: {exc}") from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise LLMError(f"Ollama JSON call timed out after {CONFIG.llm_timeout_s}s: {exc}") from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON matching the required "
                            f"schema. Error: {exc}. Reply again with ONLY valid JSON, "
                            "no markdown fences, no commentary."
                        ),
                    }
                )
    raise LLMError(f"Model did not return schema-valid JSON after {max_retries + 1} attempt(s): {last_error}")
