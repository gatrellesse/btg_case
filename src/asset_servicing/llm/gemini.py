"""Gemini client: structured output and a tool-calling loop.

Thin on purpose. There is no orchestration framework here — for eight
single-page notices, an abstraction layer is something you would have to
defend in a live session without it buying anything.
"""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, ValidationError

from .base import LLMResponse, MalformedOutput, TransientError
from .cache import ResponseCache

#: Overridable, because model ids move faster than this repository does.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "deadline", "unavailable")


class GeminiClient:
    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        cache: ResponseCache | None = None,
        offline: bool = False,
    ) -> None:
        self.model = model
        self.cache = cache
        self.offline = offline
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        self._client = None

    def available(self) -> bool:
        if self.offline:
            return False
        if not self._api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- retry ------------------------------------------------------------

    @staticmethod
    def _classify_exception(exc: Exception) -> Exception:
        text = f"{type(exc).__name__} {exc}".lower()
        if any(marker in text for marker in _TRANSIENT_MARKERS):
            return TransientError(str(exc))
        return exc

    def _call_with_backoff(self, fn, attempts: int = 3):
        """Only for transient failures: the input has not changed, and the
        call never completed, so the identical request is the right retry."""
        delay = 1.0
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                classified = self._classify_exception(exc)
                if not isinstance(classified, TransientError):
                    raise classified from exc
                last = classified
                time.sleep(delay)
                delay *= 2
        raise last if last else RuntimeError("retry exhausted")

    # -- structured output -------------------------------------------------

    def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_repair_attempts: int = 2,
    ) -> LLMResponse:
        cache_key = None
        if self.cache:
            cache_key = ResponseCache.key(self.model, prompt, system, schema.__name__)
            hit = self.cache.get(cache_key)
            if hit is not None:
                return LLMResponse(
                    text=hit.get("text", ""),
                    parsed=schema.model_validate(hit["parsed"]),
                    model=self.model,
                    from_cache=True,
                )

        from google.genai import types

        client = self._get_client()
        current_prompt = prompt
        last_error: Exception | None = None

        # The retry here is legitimate only because the prompt gains the
        # validation error each round. Same prompt twice would be waste.
        for attempt in range(max_repair_attempts + 1):
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
                system_instruction=system,
            )
            response = self._call_with_backoff(
                lambda: client.models.generate_content(
                    model=self.model, contents=current_prompt, config=config
                )
            )
            raw = getattr(response, "text", "") or ""
            try:
                parsed = getattr(response, "parsed", None)
                if parsed is None:
                    parsed = schema.model_validate_json(raw)
                elif not isinstance(parsed, schema):
                    parsed = schema.model_validate(parsed)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if attempt >= max_repair_attempts:
                    break
                current_prompt = (
                    f"{prompt}\n\n"
                    f"A resposta anterior foi rejeitada pela validação de schema:\n{exc}\n"
                    f"Corrija e responda somente com o JSON válido."
                )
                continue

            if self.cache and cache_key:
                self.cache.put(
                    cache_key,
                    {"text": raw, "parsed": parsed.model_dump(mode="json"), "model": self.model},
                )
            return LLMResponse(text=raw, parsed=parsed, model=self.model)

        raise MalformedOutput(f"schema {schema.__name__} não satisfeito: {last_error}")

    # -- tool calling ------------------------------------------------------

    def with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_impls: dict,
        *,
        system: str | None = None,
        max_turns: int = 6,
    ) -> LLMResponse:
        """Let the model choose which validators to run, then explain itself.

        The tools are pure Python; the verdict is computed from what they
        return, never from the prose the model writes around them. What the
        model contributes is orchestration and a justification — and every
        call, with its arguments and its result, goes into the audit trail.
        """
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=tools)],
            temperature=0.0,
            system_instruction=system,
        )
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        calls: list[dict] = []

        for _ in range(max_turns):
            response = self._call_with_backoff(
                lambda: client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            )
            candidate = (response.candidates or [None])[0]
            if candidate is None or not candidate.content:
                break
            contents.append(candidate.content)

            requests = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
            if not requests:
                return LLMResponse(
                    text=getattr(response, "text", "") or "",
                    tool_calls=calls,
                    model=self.model,
                )

            parts = []
            for call in requests:
                args = dict(call.args or {})
                impl = tool_impls.get(call.name)
                if impl is None:
                    result = {"error": f"tool desconhecida: {call.name}"}
                else:
                    try:
                        result = impl(**args)
                    except Exception as exc:  # noqa: BLE001
                        # A failing tool is reported back to the model rather
                        # than killing the document.
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                calls.append({"name": call.name, "args": args, "result": result})
                parts.append(
                    types.Part.from_function_response(
                        name=call.name, response={"result": json.loads(json.dumps(result, default=str))}
                    )
                )
            contents.append(types.Content(role="user", parts=parts))

        return LLMResponse(text="", tool_calls=calls, model=self.model)
