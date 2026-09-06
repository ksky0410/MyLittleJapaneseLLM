from __future__ import annotations

from prepare_quality_chat_sft import is_greeting_only, is_question_context, select_candidates


def _candidate(index: int, *, question: bool, greeting: bool = False, first: bool = False) -> dict:
    return {
        "record_index": index,
        "target_index": 1 if first else 2,
        "response_token_count": 5,
        "body_token_count": 4,
        "question_context": question,
        "greeting_only": greeting,
        "first_turn": first,
    }


def test_classifies_question_and_greeting() -> None:
    assert is_question_context("今日は何をしましたか？")
    assert is_question_context("どこへ行ったの")
    assert not is_question_context("今日は雨でした")
    assert is_greeting_only("こんにちは！")
    assert not is_greeting_only("こんにちは。今日は元気です")


def test_selection_is_deterministic_and_respects_caps() -> None:
    candidates = [
        _candidate(index, question=index % 2 == 0, greeting=index == 0, first=index == 1)
        for index in range(20)
    ]
    first = select_candidates(
        candidates,
        40,
        7,
        question_token_fraction=0.5,
        max_greeting_token_fraction=0.02,
        max_first_turn_token_fraction=0.05,
    )
    second = select_candidates(
        candidates,
        40,
        7,
        question_token_fraction=0.5,
        max_greeting_token_fraction=0.02,
        max_first_turn_token_fraction=0.05,
    )
    assert first == second
    assert sum(item["response_token_count"] for item in first) >= 40
    assert sum(item["response_token_count"] for item in first if item["greeting_only"]) <= 0
    assert sum(item["response_token_count"] for item in first if item["first_turn"]) <= 0
