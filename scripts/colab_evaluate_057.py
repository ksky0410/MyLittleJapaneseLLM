"""実験057の2条件をColab CUDAでdomain/chat評価する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/small_llm_057")
CONFIG = "configs/issue1-056base-rehearsal-ratio050-eos-ablation-colab-3k.toml"
EVALUATE = PROJECT / "scripts/evaluate_torch.py"
OUTPUT_ROOT = PROJECT / "artifacts/evaluations"
CONDITIONS = (
    "issue1-056base-rehearsal-ratio050-eos100-colab-3k",
    "issue1-056base-rehearsal-ratio050-eos050-colab-3k",
)


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    python = sys.executable
    for condition in CONDITIONS:
        checkpoint = f"artifacts/checkpoints/{condition}/best.pt"
        prefix = condition
        run(
            [
                python, str(EVALUATE), "domains",
                "--config", CONFIG, "--checkpoint", checkpoint,
                "--device", "cuda",
                "--domain", "general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin",
                "--domain", "conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin",
                "--domain", "medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin",
                "--domain", "rpc=artifacts/tokens/issue1-real-persona-chat-validation.bin",
                "--domain", "mrmp=artifacts/tokens/issue1-mrmp-validation.bin",
                "--eval-batches", "20",
                "--output", str((OUTPUT_ROOT / f"{prefix}-domains.json").relative_to(PROJECT)),
            ]
        )
        run(
            [
                python, str(EVALUATE), "chat",
                "--config", CONFIG, "--checkpoint", checkpoint,
                "--device", "cuda",
                "--input", "artifacts/corpus/conversation-v1/test.jsonl",
                "--selection-file", "experiments/evaluation/chat-test-v1.json",
                "--examples", "48", "--max-new-tokens", "64", "--seed", "42",
                "--output", str((OUTPUT_ROOT / f"{prefix}-chat-test-v1.json").relative_to(PROJECT)),
                "--text-output", str((OUTPUT_ROOT / f"{prefix}-chat-test-v1.txt").relative_to(PROJECT)),
            ]
        )


if __name__ == "__main__":
    main()
