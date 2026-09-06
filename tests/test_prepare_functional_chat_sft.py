from prepare_functional_chat_sft import (
    SOURCE_CATEGORY_FRACTIONS,
    select_candidates,
    validate_source_fractions,
)


def _candidate(index: int, category: str, *, first: bool = False) -> dict:
    return {
        "record_index": index,
        "target_index": 1,
        "response_token_count": 5,
        "body_token_count": 4,
        "category": category,
        "first_turn": first,
    }


def test_source_fraction_tables_are_complete() -> None:
    assert set(SOURCE_CATEGORY_FRACTIONS) == {"rpc", "mrmp"}
    for source in SOURCE_CATEGORY_FRACTIONS:
        assert sum(validate_source_fractions(source).values()) == 1.0


def test_selection_is_deterministic_and_reaches_target() -> None:
    candidates = [
        _candidate(index, "question_answer" if index % 2 == 0 else "other")
        for index in range(100)
    ]
    first = select_candidates(candidates, 100, 9301, "rpc")
    second = select_candidates(candidates, 100, 9301, "rpc")
    assert first == second
    assert sum(item["response_token_count"] for item in first) >= 100


def test_selection_falls_back_when_a_remaining_category_hits_turn_cap() -> None:
    candidates = [
        _candidate(index, "greeting", first=index < 3)
        for index in range(3)
    ] + [_candidate(index + 3, "other") for index in range(20)]
    selected = select_candidates(candidates, 50, 9301, "mrmp")
    assert sum(item["response_token_count"] for item in selected) >= 50
