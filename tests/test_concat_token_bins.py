from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from concat_token_bins import concat_token_bins


def test_concat_token_bins_interleaves_chunks_and_records_manifest(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    output = tmp_path / "output.bin"
    manifest = tmp_path / "manifest.json"
    np.asarray([1, 2, 3, 4, 5], dtype=np.uint32).tofile(first)
    np.asarray([10, 11, 12], dtype=np.uint32).tofile(second)

    result = concat_token_bins(
        [("first", first), ("second", second)],
        output,
        manifest,
        chunk_tokens=2,
    )

    np.testing.assert_array_equal(
        np.fromfile(output, dtype=np.uint32),
        np.asarray([1, 2, 10, 11, 3, 4, 12, 5], dtype=np.uint32),
    )
    assert result["output_token_count"] == 8
    assert json.loads(manifest.read_text())["output_sha256"] == result["output_sha256"]


def test_concat_token_bins_requires_two_inputs(tmp_path) -> None:
    source = tmp_path / "source.bin"
    np.asarray([1], dtype=np.uint32).tofile(source)
    with pytest.raises(ValueError, match="少なくとも2つ"):
        concat_token_bins(
            [("only", source)],
            tmp_path / "output.bin",
            tmp_path / "manifest.json",
            chunk_tokens=2,
        )
