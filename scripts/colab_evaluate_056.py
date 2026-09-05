"""実験056のbest checkpointをColab CUDAでdomain/chat評価する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/small_llm_056")
CONFIG = "configs/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k.toml"
CHECKPOINT = "artifacts/checkpoints/fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k/best.pt"
EVALUATE = PROJECT / "scripts/evaluate_torch.py"
OUTPUT_ROOT = PROJECT / "artifacts/evaluations"


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    python = sys.executable
    domains_output = OUTPUT_ROOT / "fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-domains.json"
    run(
        [
            python,
            str(EVALUATE),
            "domains",
            "--config",
            CONFIG,
            "--checkpoint",
            CHECKPOINT,
            "--device",
            "cuda",
            "--domain",
            "general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin",
            "--domain",
            "conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin",
            "--domain",
            "medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin",
            "--domain",
            "rpc=artifacts/tokens/issue1-real-persona-chat-validation.bin",
            "--domain",
            "mrmp=artifacts/tokens/issue1-mrmp-validation.bin",
            "--eval-batches",
            "20",
            "--output",
            str(domains_output.relative_to(PROJECT)),
        ]
    )
    chat_output = OUTPUT_ROOT / "fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-chat-test-v1.json"
    chat_text = OUTPUT_ROOT / "fineweb2-wikipedia-mid-ja-20m-swiglu-rope-torch-colab-10k-chat-test-v1.txt"
    run(
        [
            python,
            str(EVALUATE),
            "chat",
            "--config",
            CONFIG,
            "--checkpoint",
            CHECKPOINT,
            "--device",
            "cuda",
            "--input",
            "artifacts/corpus/conversation-v1/test.jsonl",
            "--selection-file",
            "experiments/evaluation/chat-test-v1.json",
            "--examples",
            "48",
            "--max-new-tokens",
            "64",
            "--seed",
            "42",
            "--output",
            str(chat_output.relative_to(PROJECT)),
            "--text-output",
            str(chat_text.relative_to(PROJECT)),
        ]
    )


if __name__ == "__main__":
    main()
