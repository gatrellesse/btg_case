"""LLM interface and the typed failures that drive retry policy.

The governing rule for the whole pipeline: **only re-run when something in the
input changed.** Each error type below exists because it answers that question
differently — a transient failure never reached the model, a malformed output
can be re-prompted with the error appended, and genuine ambiguity has nothing
left to change and must escalate instead of looping.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LLMError(Exception):
    """Base class. Nothing catches this directly."""


class TransientError(LLMError):
    """Rate limit, 5xx, timeout — the call never completed.

    Nothing about the request was wrong, so retrying the identical input is
    the correct response.
    """


class MalformedOutput(LLMError):
    """The model answered, but not in the shape asked for.

    Retrying is only justified because the prompt *changes*: the validation
    error is appended so the second attempt has information the first lacked.
    """


class LLMResponse(BaseModel):
    text: str = ""
    parsed: Any = None
    tool_calls: list[dict] = []
    usage: dict[str, int] = {}
    model: str = ""
    from_cache: bool = False
