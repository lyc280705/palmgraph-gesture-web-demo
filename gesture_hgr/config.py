from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def expand_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    return Path(os.path.expanduser(str(path))).resolve()


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    path = expand_path(path)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    path = expand_path(path)
    if path is None or not path.exists():
        raise FileNotFoundError(f'JSON not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
