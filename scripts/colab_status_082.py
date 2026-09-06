"""実験082のColab側状態を読み取り専用で確認する。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT = Path("/content/small_llm_082")
OUTPUT = PROJECT / "artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-3k"


def main() -> None:
    result: dict[str, object] = {
        "project_exists": PROJECT.exists(),
        "project_entries": sorted(path.name for path in PROJECT.iterdir()) if PROJECT.exists() else [],
        "output_exists": OUTPUT.exists(),
        "output_entries": sorted(path.name for path in OUTPUT.iterdir()) if OUTPUT.exists() else [],
    }
    metrics = OUTPUT / "metrics.jsonl"
    if metrics.is_file():
        lines = metrics.read_text().splitlines()
        result["metrics_lines"] = len(lines)
        result["metrics_tail"] = lines[-3:]
    result["processes"] = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etime,command | grep -E 'train_sft_torch|082' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
