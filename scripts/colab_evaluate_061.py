"""実験061のrehearsal ratio比較をColab CUDAでdomain/chat評価する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/small_llm_061")
CONFIG = "configs/issue1-056base-rehearsal-ratio050-eos050-colab-6k.toml"
EVALUATE = PROJECT / "scripts/evaluate_torch.py"
OUTPUT_ROOT = PROJECT / "artifacts/evaluations"
CONDITIONS = (
    "issue1-056base-rehearsal-ratio025-eos050-colab-6k-fixedlr3k",
    "issue1-056base-rehearsal-ratio075-eos050-colab-6k-fixedlr3k",
)


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    python = sys.executable
    for condition in CONDITIONS:
        checkpoint = f"artifacts/checkpoints/{condition}/best.pt"
        run([
            python, str(EVALUATE), "domains",
            "--config", CONFIG, "--checkpoint", checkpoint, "--device", "cuda",
            "--domain", "general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin",
            "--domain", "conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin",
            "--domain", "medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin",
            "--domain", "rpc=artifacts/tokens/issue1-real-persona-chat-validation.bin",
            "--domain", "mrmp=artifacts/tokens/issue1-mrmp-validation.bin",
            "--eval-batches", "20",
            "--output", str((OUTPUT_ROOT / f"{condition}-domains.json").relative_to(PROJECT)),
        ])
        run([
            python, str(EVALUATE), "chat",
            "--config", CONFIG, "--checkpoint", checkpoint, "--device", "cuda",
            "--input", "artifacts/corpus/conversation-v1/test.jsonl",
            "--selection-file", "experiments/evaluation/chat-test-v1.json",
            "--examples", "48", "--max-new-tokens", "64", "--seed", "42",
            "--output", str((OUTPUT_ROOT / f"{condition}-chat-test-v1.json").relative_to(PROJECT)),
            "--text-output", str((OUTPUT_ROOT / f"{condition}-chat-test-v1.txt").relative_to(PROJECT)),
        ])


if __name__ == "__main__":
    main()
