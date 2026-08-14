"""O que foi enviado ao modelo e o que ele devolveu, por chamada.

Existe porque o registro final não mostra a conversa: ele mostra o resultado.
Quando um campo sai errado, a pergunta é se o modelo recebeu o texto certo, se
respondeu outra coisa, ou se o schema o obrigou a preencher o que não estava
lá — e nenhuma dessas três se distingue olhando só o valor extraído.

Captura por proxy em volta do `Agent`, não em cada chamada: os quatro agentes
saem todos de `build_*`, então um ponto de instrumentação cobre os seis pontos
de uso em `nodes.py` e não fica dessincronizado quando um sétimo aparecer.

Desligado por padrão. A conversa carrega o documento inteiro em texto e a
página em imagem — é material de depuração, não de produção.
"""

from __future__ import annotations

import base64
import contextvars
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENABLED = False
_LOCK = threading.Lock()
_RECORDS: list[dict[str, Any]] = []

#: ContextVar e não global: o lote roda em ThreadPoolExecutor e cada thread
#: começa com um contexto próprio, então o nome do documento não vaza entre
#: workers.
_DOCUMENT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "as_trace_document", default=""
)

#: Prompt de extração carrega o documento inteiro; o viewer fica ilegível e o
#: arquivo enorme se guardarmos tudo. O corte é no meio, preservando início e
#: fim — é onde estão a instrução e a pergunta.
MAX_TEXT = 6000
THUMB_MAX_PIXELS = 900


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def enabled() -> bool:
    return _ENABLED


def set_document(name: str) -> None:
    _DOCUMENT.set(name)


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT:
        return text, False
    half = MAX_TEXT // 2
    return f"{text[:half]}\n\n[... {len(text) - MAX_TEXT} caracteres ...]\n\n{text[-half:]}", True


def _thumb(data: bytes, media_type: str) -> dict[str, Any]:
    """Miniatura do que o modelo viu. Sem ela o painel do image_classifier
    ficaria com um retângulo escrito 'imagem', que não responde nada."""
    out: dict[str, Any] = {"kind": "image", "media_type": media_type, "bytes": len(data)}
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.thumbnail((THUMB_MAX_PIXELS, THUMB_MAX_PIXELS))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=72)
        out["data_uri"] = "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("ascii")
    except Exception:
        pass  # sem miniatura o registro ainda vale; o tamanho já é informação
    return out


def _part(part: Any) -> dict[str, Any]:
    kind = getattr(part, "part_kind", type(part).__name__)
    content = getattr(part, "content", None)

    if kind == "tool-call" or hasattr(part, "tool_name") and hasattr(part, "args"):
        args = part.args
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        return {"kind": "tool-call", "tool": part.tool_name, "args": args}

    if isinstance(content, (list, tuple)):
        pieces = []
        for item in content:
            data = getattr(item, "data", None)
            if isinstance(data, (bytes, bytearray)):
                pieces.append(_thumb(bytes(data), getattr(item, "media_type", "image/png")))
            else:
                text, clipped = _clip(str(item))
                pieces.append({"kind": "text", "text": text, "clipped": clipped})
        return {"kind": kind, "pieces": pieces}

    text, clipped = _clip("" if content is None else str(content))
    return {"kind": kind, "text": text, "clipped": clipped}


def _messages(result: Any) -> list[dict[str, Any]]:
    try:
        messages = result.all_messages()
    except Exception:
        return []
    out = []
    for message in messages:
        parts = []
        # `instructions` é atributo da mensagem, não parte — sem isto o painel
        # mostraria o prompt sem as regras sob as quais ele foi respondido,
        # que é metade do que se quer auditar.
        instructions = getattr(message, "instructions", None)
        if instructions:
            text, clipped = _clip(str(instructions))
            parts.append({"kind": "system-prompt", "text": text, "clipped": clipped})
        parts.extend(_part(p) for p in getattr(message, "parts", []))
        out.append(
            {"role": getattr(message, "kind", type(message).__name__), "parts": parts}
        )
    return out


def _usage(result: Any) -> dict[str, Any]:
    """`usage` mudou de método para propriedade entre versões do pydantic-ai."""
    usage = getattr(result, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            return {}
    if usage is None:
        return {}
    return {
        k: getattr(usage, k)
        for k in ("input_tokens", "output_tokens", "requests")
        if getattr(usage, k, None) is not None
    }


def _output(result: Any) -> Any:
    output = getattr(result, "output", None)
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    return None if output is None else str(output)[:MAX_TEXT]


class _Traced:
    """Proxy: encaminha tudo para o Agent e grava o que passa por `run_sync`.

    Proxy em vez de monkeypatch da instância porque o `Agent` do pydantic-ai
    não garante atributos graváveis entre versões.
    """

    def __init__(self, agent: Any, name: str) -> None:
        self._agent = agent
        self._name = name

    def __getattr__(self, item: str) -> Any:
        return getattr(self._agent, item)

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            result = self._agent.run_sync(*args, **kwargs)
        except Exception as exc:
            _append(self._name, None, exc, time.perf_counter() - started)
            raise
        _append(self._name, result, None, time.perf_counter() - started)
        return result


def _append(name: str, result: Any, error: Exception | None, seconds: float) -> None:
    record = {
        "agent": name,
        "document": _DOCUMENT.get(),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 2),
        "exchange": _messages(result) if result is not None else [],
        "usage": _usage(result) if result is not None else {},
        "output": _output(result) if result is not None else None,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    with _LOCK:
        _RECORDS.append(record)


def wrap(agent: Any, name: str) -> Any:
    return _Traced(agent, name) if _ENABLED else agent


def records() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_RECORDS)


def dump(path: str | Path) -> int:
    data = records()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(data)
