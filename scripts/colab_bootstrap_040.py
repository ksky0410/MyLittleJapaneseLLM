"""実験040のColab bootstrapを固定configで起動する。"""

from __future__ import annotations

import sys

from colab_bootstrap_train import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--config",
        "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-5k.toml",
        "--device",
        "auto",
    ]
    main()
