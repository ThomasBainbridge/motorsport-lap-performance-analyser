from __future__ import annotations

from pathlib import Path
from typing import Any

import fastf1
import yaml
from fastf1 import Cache

from .utils import ensure_dir


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_output_dirs(base_output_dir: str | Path) -> dict[str, Path]:
    base = ensure_dir(base_output_dir)
    figures = ensure_dir(base / "figures")
    tables = ensure_dir(base / "tables")
    reports = ensure_dir(base / "reports")
    return {"base": base, "figures": figures, "tables": tables, "reports": reports}


def configure_cache(cache_dir: str | Path) -> Path:
    cache_path = ensure_dir(cache_dir)
    Cache.enable_cache(str(cache_path))
    return cache_path


def load_session_from_config(config: dict[str, Any]):
    session_cfg = config["session"]
    paths_cfg = config["paths"]

    configure_cache(paths_cfg["cache_dir"])
    session = fastf1.get_session(
        int(session_cfg["year"]),
        session_cfg["grand_prix"],
        session_cfg["session"],
    )
    session.load()
    return session
