from audit_issue1_short_prompts import audit_evaluation, scan_prompt_matches


class TokenCounter:
    def __call__(self, text: str) -> int:
        return len(text)


def test_scan_prompt_matches_records_response_and_selection() -> None:
    records = [
        {
            "conversation_id": "rpc:1",
            "source_dialogue_id": "1",
            "source_file": "1.json",
            "turns": [
                {"speaker_id": "A", "text": "まじで"},
                {"speaker_id": "B", "text": "ほんと？"},
                {"speaker_id": "A", "text": "なんかさ、明日ひま？"},
                {"speaker_id": "B", "text": "うん、ひまだよ"},
            ],
        }
    ]
    matches = scan_prompt_matches(
        records,
        "rpc",
        "train",
        ("まじで", "明日ひま？"),
        TokenCounter(),
        {("rpc", 0, 1)},
    )
    assert [item["match_type"] for item in matches] == ["exact", "substring"]
    assert matches[0]["response"] == "ほんと？"
    assert matches[0]["sft_selected"] is True
    assert matches[1]["response_function"] is not None


def test_audit_evaluation_joins_selection_metadata() -> None:
    evaluation = {
        "results": [
            {
                "conversation_id": "rpc:1",
                "record_index": 2,
                "target_index": 1,
                "source": "rpc",
                "stratum": "short",
                "history_truncated": True,
                "train_text_overlap": False,
                "prompt_token_count": 20,
                "reference_token_count": 4,
                "generated_token_count": 3,
                "token_overlap_f1": 0.5,
                "eos_reached": 1,
            }
        ]
    }
    selection = {
        "examples": [
            {
                "conversation_id": "rpc:1",
                "record_index": 2,
                "target_index": 1,
                "history_token_count": 300,
            }
        ]
    }
    result = audit_evaluation(evaluation, selection)
    assert result["count"] == 1
    assert result["rows"][0]["history_token_count"] == 300
    assert result["groups"]["rpc"]["mean_f1"] == 0.5
