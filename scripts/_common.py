"""スクリプト共通のリポジトリルート解決とsrc読込。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def repo_path(value: str | Path) -> Path:
    """相対パスを、現在のcwdではなくリポジトリルートから解決する。"""

    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
