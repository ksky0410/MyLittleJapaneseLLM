"""実験082のSFT起動エラーを標準出力・標準エラー付きで再現する。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/small_llm_082")


def main() -> None:
    command = [
        sys.executable,
        str(PROJECT / "scripts/train_sft_torch.py"),
        "--config",
        "configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml",
        "--base-checkpoint",
        "artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt",
        "--train-data",
        "artifacts/sft/issue1-both-balanced-v1/train.npz",
        "--validation-data",
        "artifacts/sft/issue1-both-full-v1/validation.npz",
        "--output-dir",
        "artifacts/checkpoints/issue1-both-sft-debug",
        "--samples-dir",
        "artifacts/samples/issue1-both-sft-debug",
        "--lr-schedule-steps",
        "1",
        "--max-steps",
        "1",
        "--eos-loss-weight",
        "0.5",
        "--rehearsal-tokens",
        "artifacts/tokens/mixed-ja-80-10-10-v2-train.bin",
        "--rehearsal-ratio",
        "0.20",
        "--sample-template",
        "conversation",
        "--sample-speaker-a",
        "DA",
        "--sample-speaker-b",
        "DC",
        "--device",
        "auto",
    ]
    completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, check=False)
    print(
        json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-12000:],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
