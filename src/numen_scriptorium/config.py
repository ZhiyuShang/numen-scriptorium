from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from numen_scriptorium.paths import ROOT


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/dict: {p}")
    return data


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged
