"""実験040のColab bootstrapを固定configで起動する。"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def _extract_bundle() -> Path:
    """colab execの一時実行ディレクトリからbundle内のコードを使えるようにする。"""

    bundle = Path("/content/small_llm_bundle.tar.gz")
    project_dir = Path("/content/small_llm")
    project_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(project_dir, filter="data")
    scripts_dir = project_dir / "scripts"
    sys.path.insert(0, str(scripts_dir))
    return project_dir


if __name__ == "__main__":
    _extract_bundle()
    from colab_bootstrap_train import main

    sys.argv = [
        sys.argv[0],
        "--config",
        "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k.toml",
        "--device",
        "auto",
    ]
    main()
