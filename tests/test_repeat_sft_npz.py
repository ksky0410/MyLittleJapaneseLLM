from __future__ import annotations

import json

import numpy as np
import pytest

from repeat_sft_npz import repeat_sft_npz


def _write_input(path, count: int = 3) -> None:
    values = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
    np.savez_compressed(
        path,
        input_ids=values,
        target_ids=values + 1,
        loss_mask=np.ones_like(values, dtype=np.uint8),
    )


def test_repeat_sft_npz_repeats_and_shuffles_deterministically(tmp_path) -> None:
    source = tmp_path / "source.npz"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    _write_input(source)

    first_result = repeat_sft_npz(
        source, first, first_manifest, repeat=4, seed=303
    )
    second_result = repeat_sft_npz(
        source, second, second_manifest, repeat=4, seed=303
    )

    with np.load(first) as first_arrays, np.load(second) as second_arrays:
        for key in ("input_ids", "target_ids", "loss_mask"):
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
    assert first_result["output_example_count"] == 12
    assert first_result["output_response_token_count"] == 48
    assert first_result["output_sha256"] == second_result["output_sha256"]
    first_manifest_data = json.loads(first_manifest.read_text())
    second_manifest_data = json.loads(second_manifest.read_text())
    first_manifest_data["output_path"] = second_manifest_data["output_path"]
    assert first_manifest_data == second_manifest_data


def test_repeat_sft_npz_rejects_zero_repeat(tmp_path) -> None:
    source = tmp_path / "source.npz"
    _write_input(source)
    with pytest.raises(ValueError, match="repeatは1以上"):
        repeat_sft_npz(source, tmp_path / "out.npz", tmp_path / "out.json", repeat=0, seed=1)
