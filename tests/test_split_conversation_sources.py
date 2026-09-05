import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split_conversation_sources import export_sources


def _write_split(root: Path, split: str) -> None:
    records = [
        {
            "conversation_id": f"rpc:{split}",
            "dataset": "real-persona-chat",
            "turns": [
                {"speaker_id": "A", "text": "こんにちは"},
                {"speaker_id": "B", "text": "元気です"},
            ],
        },
        {
            "conversation_id": f"mrmp:{split}",
            "dataset": "mrmp",
            "turns": [{"speaker_id": "C", "text": "それな"}],
        },
    ]
    (root / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_exports_each_dataset_as_complete_conversation_blocks(tmp_path: Path) -> None:
    source = tmp_path / "conversation"
    source.mkdir()
    _write_split(source, "train")

    manifest = export_sources(source, tmp_path / "out", ("train",))

    rpc = (tmp_path / "out/real-persona-chat-train.txt").read_text(encoding="utf-8")
    mrmp = (tmp_path / "out/mrmp-train.txt").read_text(encoding="utf-8")
    assert "<|speaker:A|>こんにちは" in rpc
    assert "<|speaker:B|>元気です" in rpc
    assert rpc.count("<|startofconversation|>") == 1
    assert mrmp.count("<|startofconversation|>") == 1
    assert manifest["outputs"]["real-persona-chat:train"]["conversation_count"] == 1
    assert manifest["outputs"]["mrmp:train"]["turn_count"] == 1
