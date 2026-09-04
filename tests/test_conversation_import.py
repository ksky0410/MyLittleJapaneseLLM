from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_conversations import import_conversations


class ConversationImportTests(unittest.TestCase):
    def _write_conversation(
        self,
        path: Path,
        dialogue_id: int | str,
        turns: list[dict[str, object]],
        *,
        mrmp: bool = False,
    ) -> None:
        payload: dict[str, object] = {
            "dialogue_id": dialogue_id,
            "interlocutors": ["A", "B"],
            "utterances": turns,
            "evaluations": [{"interlocutor_id": "A", "satisfaction": 5}],
        }
        if mrmp:
            payload.update(
                {
                    "dialogue_type": "First time",
                    "relationship": [],
                }
            )
            for turn in turns:
                turn["mention_to"] = ["B"]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_imports_both_json_layouts_and_keeps_conversation_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_root = root / "real-persona-chat" / "real_persona_chat" / "dialogues"
            mrmp_root = (
                root
                / "multi-relational-multi-party-chat-corpus"
                / "multi_relational_multi_party_chat_corpus"
                / "dialogues"
                / "A_first_time"
            )
            real_root.mkdir(parents=True)
            mrmp_root.mkdir(parents=True)
            self._write_conversation(
                real_root / "00001.json",
                1,
                [
                    {
                        "utterance_id": 1,
                        "interlocutor_id": "B",
                        "text": "二番目の発話",
                        "timestamp": "2025-01-01T00:00:00",
                    },
                    {
                        "utterance_id": 0,
                        "interlocutor_id": "A",
                        "text": "最初の発話",
                        "timestamp": "2025-01-01T00:00:00",
                    },
                ],
            )
            self._write_conversation(
                real_root / "00002.json",
                2,
                [{"utterance_id": 0, "interlocutor_id": "A", "text": "三番目の会話"}],
            )
            self._write_conversation(
                mrmp_root / "A00101.json",
                "A00101",
                [{"utterance_id": 0, "interlocutor_id": "A", "text": "MRMPの発話"}],
                mrmp=True,
            )
            output = root / "conversation-v1"
            manifest = import_conversations(
                [
                    root / "real-persona-chat",
                    root / "multi-relational-multi-party-chat-corpus",
                ],
                output,
                source_names=["real-persona-chat", "mrmp"],
                source_urls=[
                    "https://github.com/nu-dialogue/real-persona-chat",
                    "https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus",
                ],
                source_commits=["a" * 40, "b" * 40],
                seed=7,
                validation_ratio=0.25,
                test_ratio=0.25,
            )

            self.assertEqual(manifest["format"], "conversation-import-v1")
            self.assertEqual(manifest["conversation_count"], 3)
            self.assertEqual(manifest["turn_count"], 4)
            self.assertEqual(
                {
                    split: values["conversation_count"]
                    for split, values in manifest["splits"].items()
                },
                {"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(manifest["sources"][0]["commit_sha"], "a" * 40)
            self.assertEqual(manifest["sources"][1]["commit_sha"], "b" * 40)
            self.assertEqual(manifest["sources"][0]["input_file_count"], 2)
            self.assertEqual(
                manifest["sources"][0]["input_files"][0]["sha256"],
                hashlib.sha256((real_root / "00001.json").read_bytes()).hexdigest(),
            )

            records = []
            for split in ("train", "validation", "test"):
                records.extend(
                    json.loads(line)
                    for line in (output / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            self.assertEqual(
                {record["conversation_id"] for record in records},
                {
                    "real-persona-chat:real_persona_chat/dialogues/00001.json",
                    "real-persona-chat:real_persona_chat/dialogues/00002.json",
                    "mrmp:multi_relational_multi_party_chat_corpus/dialogues/A_first_time/A00101.json",
                },
            )
            first = next(
                record for record in records if record["source_dialogue_id"] == "1"
            )
            self.assertEqual(
                [turn["text"] for turn in first["turns"]],
                ["最初の発話", "二番目の発話"],
            )
            self.assertNotIn("timestamp", json.dumps(records, ensure_ascii=False))
            self.assertNotIn("evaluations", json.dumps(records, ensure_ascii=False))
            self.assertNotIn("mention_to", json.dumps(records, ensure_ascii=False))

            text = "\n".join(
                (output / f"{split}.txt").read_text(encoding="utf-8")
                for split in ("train", "validation", "test")
            )
            self.assertIn("<|speaker:A|>", text)
            self.assertIn("<|speaker:B|>", text)
            self.assertNotIn("https://", text)

    def test_same_seed_reproduces_assignment_and_split_is_conversation_level(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogues = root / "source" / "dialogues"
            dialogues.mkdir(parents=True)
            for number in range(1, 11):
                self._write_conversation(
                    dialogues / f"{number:05d}.json",
                    number,
                    [
                        {
                            "utterance_id": 0,
                            "interlocutor_id": "A",
                            "text": f"会話{number}",
                        }
                    ],
                )
            first_output = root / "first"
            second_output = root / "second"
            first = import_conversations([root / "source"], first_output, seed=123)
            second = import_conversations([root / "source"], second_output, seed=123)
            self.assertEqual(
                {
                    split: [record["conversation_id"] for record in values]
                    for split, values in (
                        (
                            split,
                            [
                                json.loads(line)
                                for line in (first_output / f"{split}.jsonl")
                                .read_text(encoding="utf-8")
                                .splitlines()
                            ],
                        )
                        for split in ("train", "validation", "test")
                    )
                },
                {
                    split: [record["conversation_id"] for record in values]
                    for split, values in (
                        (
                            split,
                            [
                                json.loads(line)
                                for line in (second_output / f"{split}.jsonl")
                                .read_text(encoding="utf-8")
                                .splitlines()
                            ],
                        )
                        for split in ("train", "validation", "test")
                    )
                },
            )
            self.assertEqual(first["seed"], 123)
            self.assertEqual(second["seed"], 123)
            assignments = []
            for split in ("train", "validation", "test"):
                assignments.extend(
                    json.loads(line)["conversation_id"]
                    for line in (first_output / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            self.assertEqual(len(assignments), len(set(assignments)))


if __name__ == "__main__":
    unittest.main()
