"""Configuration loading, shared by every layer.

Lives at the root rather than inside a layer because all four read from
``config/``: classification takes its markers, extraction its anchors,
validation its thresholds. Nothing sample-specific is hard-coded anywhere —
the YAML is the only place a vocabulary appears.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=None)
def load_config(name: str, config_dir: str | None = None) -> dict:
    path = Path(config_dir or CONFIG_DIR) / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def compile_marker(pattern: str) -> re.Pattern:
    """Compile a config pattern so it survives OCR.

    Every literal space becomes ``\\s*``: scanned text loses spaces constantly
    ("Jurossobreo.CapitalProprio"), and a marker that insists on them would
    silently fail on exactly the documents that need help most.
    """
    return re.compile(pattern.replace(" ", r"\s*"), re.IGNORECASE)

