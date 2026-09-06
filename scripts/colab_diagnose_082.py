"""実験082のColab失敗原因を読み取り中心に調べる。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/small_llm_082")
TRAIN = PROJECT / "scripts/train_sft_torch.py"


def main() -> None:
    result: dict[str, object] = {
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "train_exists": TRAIN.is_file(),
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["cuda_device_count"] = torch.cuda.device_count()
        result["mps_available"] = torch.backends.mps.is_available()
    except Exception as exc:  # pragma: no cover - remote diagnostic
        result["torch_import_error"] = repr(exc)
    command = [
        sys.executable,
        str(TRAIN),
        "--help",
    ]
    completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, check=False)
    result["help_returncode"] = completed.returncode
    result["help_stdout"] = completed.stdout[-4000:]
    result["help_stderr"] = completed.stderr[-4000:]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
