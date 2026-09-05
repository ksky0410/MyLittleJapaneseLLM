"""実験060のColab評価用ディレクトリを作成する。"""

from pathlib import Path


ROOT = Path("/content/small_llm_060")


def main() -> None:
    for relative in (
        "artifacts/corpus/conversation-v1",
        "experiments/evaluation",
        "artifacts/evaluations",
    ):
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        print(path)


if __name__ == "__main__":
    main()
