"""Colab VMへbundleを展開し、PyTorch学習スクリプトを起動する。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="/content/small_llm_bundle.tar.gz")
    parser.add_argument("--project-dir", default="/content/small_llm")
    parser.add_argument("--config", default="configs/fineweb2-mixed-ja-20m-torch-smoke.toml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    project_dir = Path(args.project_dir)
    if not bundle.is_file():
        raise FileNotFoundError(f"bundleが見つかりません: {bundle}")
    project_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(project_dir, filter="data")

    command = [
        sys.executable,
        str(project_dir / "scripts" / "train_torch.py"),
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.no_amp:
        command.append("--no-amp")
    subprocess.run(command, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()
