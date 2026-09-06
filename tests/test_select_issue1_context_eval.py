from select_issue1_context_eval import select_candidates


class Processor:
    def encode(self, text: str, out_type=int):
        return list(range(len(text)))

    def eos_id(self):
        return 0


def test_select_candidates_separates_surface_and_semantic_rules() -> None:
    records = [
        {
            "conversation_id": "rpc:1",
            "source": "real-persona-chat",
            "turns": [
                {"speaker_id": "A", "text": "今日は何してましたか！"},
                {"speaker_id": "B", "text": "仕事でした"},
            ],
        },
        {
            "conversation_id": "mrmp:1",
            "source": "mrmp",
            "turns": [
                {"speaker_id": "A", "text": "それなー！"},
                {"speaker_id": "B", "text": "ほんとそれ"},
            ],
        },
    ]
    selected, counts = select_candidates(records, Processor(), seed=1, per_source=2)
    assert any(item["category"] == "today-activity" and item["match_type"] == "surface" for item in selected)
    assert any(item["category"] == "casual-agreement" and item["match_type"] == "surface" for item in selected)
    assert counts["real-persona-chat:today-activity"]["available"] == 1
