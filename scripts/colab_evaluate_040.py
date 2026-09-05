"""実験040のPyTorch checkpointをColab上でdomain/chat評価する。"""

from __future__ import annotations

import json
import sys
import tarfile
from argparse import Namespace
from pathlib import Path


def _extract_bundle() -> Path:
    bundle = Path("/content/small_llm_eval_040.tar.gz")
    project_dir = Path("/content/small_llm")
    project_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(project_dir, filter="data")
    sys.path.insert(0, str(project_dir / "scripts"))
    return project_dir


def main() -> None:
    _extract_bundle()
    from evaluate_torch import evaluate_chat, evaluate_domains

    config = "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k.toml"
    checkpoint = (
        "artifacts/checkpoints/"
        "fineweb2-wikipedia-mid-ja-20m-torch-colab-5k/step_004900.pt"
    )
    domains = Namespace(
        config=config,
        checkpoint=checkpoint,
        device="auto",
        no_amp=False,
        eval_batches=20,
        output="artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-domains.json",
        domain=[
            ("general", "artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin"),
            ("conversation", "artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin"),
            ("medical", "artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin"),
            ("fineweb", "artifacts/tokens/fineweb2-edu-japanese-v1-test.bin"),
            ("wikipedia", "artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin"),
        ],
    )
    domain_result = evaluate_domains(domains)
    print(json.dumps(domain_result, ensure_ascii=False, indent=2))

    chat = Namespace(
        config=config,
        checkpoint=checkpoint,
        device="auto",
        no_amp=False,
        input="artifacts/corpus/conversation-v1/test.jsonl",
        output="artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-chat-test-v1.json",
        text_output="artifacts/evaluations/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k-chat-test-v1.txt",
        examples=48,
        max_new_tokens=64,
        seed=42,
        selection_file="experiments/evaluation/chat-test-v1.json",
    )
    chat_result = evaluate_chat(chat)
    print(json.dumps(chat_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
