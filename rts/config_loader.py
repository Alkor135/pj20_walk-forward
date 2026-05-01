from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _apply_placeholders(settings: dict[str, Any], *, model_dir: str | None = None) -> dict[str, Any]:
    ticker = str(settings.get("ticker", ""))
    ticker_lc = str(settings.get("ticker_lc", ticker.lower()))
    placeholders = {
        "{ticker}": ticker,
        "{ticker_lc}": ticker_lc,
        "{model_dir}": model_dir or "",
    }
    out = deepcopy(settings)
    for key, value in list(out.items()):
        if isinstance(value, str):
            for old, new in placeholders.items():
                value = value.replace(old, new)
            out[key] = value
    return out


def ticker_dir_from_script(script_file: str | Path) -> Path:
    script_dir = Path(script_file).resolve().parent
    if script_dir.name in {"combine", "shared"}:
        return script_dir.parent
    return script_dir.parent


def load_ticker_config(script_file: str | Path) -> dict[str, Any]:
    ticker_dir = ticker_dir_from_script(script_file)
    settings_path = ticker_dir / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.yaml не найден: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}


def load_settings_for(script_file: str | Path, section: str) -> dict[str, Any]:
    script_dir = Path(script_file).resolve().parent
    raw = load_ticker_config(script_file)
    common = raw.get("common") or {}

    if section == "model":
        model_dir = script_dir.name
        settings = _deep_merge(common, raw.get("model_defaults") or {})
        settings = _deep_merge(settings, (raw.get("models") or {}).get(model_dir) or {})
        settings["model_dir"] = model_dir
        return _apply_placeholders(settings, model_dir=model_dir)

    settings = _deep_merge(common, raw.get(section) or {})
    return _apply_placeholders(settings)


def load_model_settings(ticker_dir: Path, model_dir: str) -> dict[str, Any]:
    raw = yaml.safe_load((ticker_dir / "settings.yaml").read_text(encoding="utf-8")) or {}
    common = raw.get("common") or {}
    settings = _deep_merge(common, raw.get("model_defaults") or {})
    settings = _deep_merge(settings, (raw.get("models") or {}).get(model_dir) or {})
    settings["model_dir"] = model_dir
    return _apply_placeholders(settings, model_dir=model_dir)
