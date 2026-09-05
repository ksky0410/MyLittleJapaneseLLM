"""実験047の20M本学習をColab T4で固定設定から起動する。"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def main() -> None:
    bundle = Path("/content/exp047_bundle.tar.gz")
    project_dir = Path("/content/small_llm_047_full")
    if not bundle.is_file():
        raise FileNotFoundError(f"bundleが見つかりません: {bundle}")
    project_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(project_dir, filter="data")
    sys.path.insert(0, str(project_dir / "scripts"))
    from colab_bootstrap_train import main as train_main

    sys.argv = [
        sys.argv[0],
        "--bundle",
        str(bundle),
        "--project-dir",
        str(project_dir),
        "--config",
        "configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-2p5k.toml",
        "--device",
        "auto",
    ]
    train_main()


if __name__ == "__main__":
    main()
