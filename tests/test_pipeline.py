from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.corpus import (
    normalize_line,
    read_documents,
    split_documents,
    write_split,
)
from my_little_japanese_llm.data import evaluation_batches, make_batch
from my_little_japanese_llm.model import (
    MLX_AVAILABLE,
    TinyJapaneseGPT,
    estimate_parameter_count,
)
from my_little_japanese_llm.tokenizer import (
    DEFAULT_MAX_SENTENCE_LENGTH,
    encode_text_file,
    load_processor,
    train_sentencepiece,
)
from my_little_japanese_llm.training import (
    load_checkpoint,
    save_checkpoint,
    signature_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_MLX = MLX_AVAILABLE


class NumpyMX:
    """data.pyのbatch形状をMLXなしでも確認するための小さな代替。"""

    @staticmethod
    def array(value):
        return np.asarray(value)


class PipelineTests(unittest.TestCase):
    def test_config_and_repo_relative_paths(self) -> None:
        config = load_config(ROOT / "configs/debug.toml")
        self.assertEqual(config.model.dim, 64)
        self.assertEqual(config.training.max_steps, 100)
        self.assertIsNone(config.training.checkpoint_interval)
        self.assertEqual(config.paths.train_tokens, ROOT / "artifacts/tokens/train.bin")

    def test_colab_long_run_can_separate_checkpoint_interval(self) -> None:
        config = load_config(
            ROOT / "configs/fineweb2-wikipedia-mid-ja-20m-torch-colab-10k.toml"
        )
        self.assertEqual(config.training.eval_interval, 100)
        self.assertEqual(config.training.sample_interval, 100)
        self.assertEqual(config.training.checkpoint_interval, 1000)

    def test_aozora_5m_config_targets_formal_corpus_and_model_size(self) -> None:
        config = load_config(ROOT / "configs/aozora-5m.toml")
        self.assertEqual(
            config.paths.prepared_dir,
            ROOT / "artifacts/corpus/aozora-neko-formal-v2",
        )
        self.assertEqual(
            config.paths.tokenizer_model,
            ROOT / "artifacts/tokenizer/aozora-neko-formal-v2-unigram.model",
        )
        self.assertEqual(config.model.context_length, 256)
        parameters = estimate_parameter_count(
            4096,
            config.model.dim,
            config.model.layers,
            config.model.heads,
            config.model.context_length,
            config.model.mlp_ratio,
        )
        self.assertGreaterEqual(parameters, 4_500_000)
        self.assertLessEqual(parameters, 5_500_000)

    def test_fineweb2_20m_config_changes_only_capacity(self) -> None:
        config = load_config(ROOT / "configs/fineweb2-mixed-ja-20m-2p5k.toml")
        self.assertEqual(config.model.dim, 384)
        self.assertEqual(config.model.layers, 10)
        self.assertEqual(config.model.heads, 6)
        self.assertEqual(config.model.context_length, 256)
        self.assertEqual(config.model.position_embedding, "absolute")
        parameters = estimate_parameter_count(
            4096,
            config.model.dim,
            config.model.layers,
            config.model.heads,
            config.model.context_length,
            config.model.mlp_ratio,
            config.model.position_embedding,
        )
        self.assertEqual(parameters, 19_382_016)

    def test_wikipedia_augmentation_keeps_5m_model_and_uses_new_paths(self) -> None:
        config = load_config(
            ROOT / "configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml"
        )
        self.assertEqual(config.model.dim, 240)
        self.assertEqual(config.model.layers, 6)
        self.assertEqual(config.model.context_length, 256)
        self.assertEqual(config.training.max_steps, 2500)
        self.assertIn("wikipedia", str(config.paths.train_tokens))
        self.assertIn("wikipedia", str(config.paths.checkpoint_dir))
        parameters = estimate_parameter_count(
            4096,
            config.model.dim,
            config.model.layers,
            config.model.heads,
            config.model.context_length,
            config.model.mlp_ratio,
            config.model.position_embedding,
        )
        self.assertEqual(parameters, 5_197_920)

    def test_position_embedding_defaults_to_absolute_and_rope_config_is_valid(
        self,
    ) -> None:
        debug = load_config(ROOT / "configs/debug.toml")
        rope = load_config(ROOT / "configs/rope-mixed-ja-5m-smoke.toml")
        self.assertEqual(debug.model.position_embedding, "absolute")
        self.assertEqual(debug.model.norm_type, "layernorm")
        self.assertEqual(rope.model.position_embedding, "rope")
        absolute = estimate_parameter_count(4096, 240, 6, 6, 256, 4, "absolute")
        rope_parameters = estimate_parameter_count(4096, 240, 6, 6, 256, 4, "rope")
        self.assertEqual(absolute - rope_parameters, 256 * 240)

        rmsnorm = replace(debug.model, norm_type="rmsnorm")
        rmsnorm.validate()
        layernorm_parameters = estimate_parameter_count(
            4096, 240, 6, 6, 256, 4, "rope", "layernorm"
        )
        rmsnorm_parameters = estimate_parameter_count(
            4096, 240, 6, 6, 256, 4, "rope", "rmsnorm"
        )
        self.assertEqual(layernorm_parameters - rmsnorm_parameters, 13 * 240)
        with self.assertRaisesRegex(ValueError, "norm_type"):
            replace(debug.model, norm_type="unknown").validate()

    def test_corpus_normalization_and_deterministic_split(self) -> None:
        self.assertEqual(normalize_line("  ＡＩ\tの  実験  "), "AI の 実験")
        documents = [f"文書{i}" for i in range(10)]
        first = split_documents(documents, 0.2, 42)
        second = split_documents(documents, 0.2, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)

    def test_corpus_manifest_contains_hash_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("猫が歩く。\n犬が走る。\n", encoding="utf-8")
            train, val = split_documents(read_documents(source), 0.5, 7)
            manifest = write_split(root / "split", train, val, source, 0.5, 7)
            loaded = json.loads(
                (root / "split/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded, manifest)
            self.assertEqual(
                loaded["train_documents"] + loaded["validation_documents"], 2
            )
            self.assertEqual(len(loaded["source_sha256"]), 64)

    def test_data_batch_has_shifted_targets_and_short_data_error(self) -> None:
        tokens = np.arange(20, dtype=np.int32)
        inputs, targets = make_batch(tokens, 3, 5, np.random.default_rng(1), NumpyMX())
        self.assertEqual(inputs.shape, (3, 5))
        np.testing.assert_array_equal(targets, inputs + 1)
        batches = evaluation_batches(tokens, 2, 5, 3, NumpyMX())
        self.assertEqual(sum(batch[0].shape[0] for batch in batches), 3)
        with self.assertRaisesRegex(ValueError, "context_length"):
            make_batch(
                np.arange(5, dtype=np.int32), 1, 5, np.random.default_rng(1), NumpyMX()
            )

    @unittest.skipUnless(
        importlib.util.find_spec("sentencepiece"), "SentencePiece未導入"
    )
    def test_sentencepiece_handles_small_corpus_and_eos_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("猫が歩く。\n犬が走る。\n鳥が飛ぶ。\n", encoding="utf-8")
            model_path, _, effective = train_sentencepiece(
                source, root / "ja", 32, "unigram"
            )
            processor = load_processor(model_path)
            ids = encode_text_file(model_path, source)
            self.assertGreaterEqual(effective, 16)
            self.assertGreaterEqual(processor.vocab_size(), 4)
            self.assertEqual(ids.count(processor.eos_id()), 3)

    def test_sentencepiece_accepts_utf8_byte_length_override(self) -> None:
        class RecordingTrainer:
            call: ClassVar[dict] = {}

            @classmethod
            def train(cls, **kwargs):
                cls.call = kwargs

        fake_sentencepiece = SimpleNamespace(SentencePieceTrainer=RecordingTrainer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.txt"
            source.write_text("日本語の長い文です。\n", encoding="utf-8")
            with patch(
                "my_little_japanese_llm.tokenizer.require_sentencepiece",
                return_value=fake_sentencepiece,
            ):
                train_sentencepiece(
                    source,
                    root / "ja",
                    32,
                    max_sentence_length=20_000,
                )

            self.assertEqual(RecordingTrainer.call["max_sentence_length"], 20_000)
            self.assertEqual(DEFAULT_MAX_SENTENCE_LENGTH, 4192)
            with self.assertRaisesRegex(ValueError, "max_sentence_length"):
                train_sentencepiece(
                    source,
                    root / "invalid",
                    32,
                    max_sentence_length=0,
                )

    def test_model_parameter_estimate_is_positive(self) -> None:
        self.assertGreater(estimate_parameter_count(128, 64, 2, 4, 64), 100_000)

    @unittest.skipUnless(HAS_MLX, "MLX未導入")
    def test_mlx_forward_shape(self) -> None:
        import mlx.core as mx

        model = TinyJapaneseGPT(32, 16, 2, 4, 8, 2)
        logits = model(mx.array([[1, 2, 3, 4], [4, 3, 2, 1]]))
        mx.eval(logits)
        self.assertEqual(tuple(logits.shape), (2, 4, 32))

    @unittest.skipUnless(HAS_MLX, "MLX未導入")
    def test_mlx_rope_forward_shape(self) -> None:
        import mlx.core as mx

        model = TinyJapaneseGPT(32, 16, 2, 4, 8, 2, "rope")
        logits = model(mx.array([[1, 2, 3, 4], [4, 3, 2, 1]]))
        mx.eval(logits)
        self.assertEqual(tuple(logits.shape), (2, 4, 32))

    @unittest.skipUnless(HAS_MLX, "MLX未導入")
    def test_mlx_rmsnorm_forward_shape(self) -> None:
        import mlx.core as mx

        model = TinyJapaneseGPT(32, 16, 2, 4, 8, 2, "rope", "rmsnorm")
        logits = model(mx.array([[1, 2, 3, 4], [4, 3, 2, 1]]))
        mx.eval(logits)
        self.assertEqual(tuple(logits.shape), (2, 4, 32))

    @unittest.skipUnless(HAS_MLX, "MLX未導入")
    def test_checkpoint_metadata_is_checked_before_load(self) -> None:
        import mlx.core as mx

        config = load_config(ROOT / "configs/debug.toml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_000001.npz"
            model = TinyJapaneseGPT(32, 16, 1, 4, 8, 2)
            mx.eval(model(mx.array([[1, 2, 3, 4]])))
            signature = signature_from_config(config, 32)
            signature.update(
                {
                    "dim": 16,
                    "layers": 1,
                    "heads": 4,
                    "context_length": 8,
                    "mlp_ratio": 2,
                }
            )
            save_checkpoint(model, path, signature, {"step": 1})
            restored = TinyJapaneseGPT(32, 16, 1, 4, 8, 2)
            metadata = load_checkpoint(restored, path, signature)
            self.assertEqual(metadata["model"], signature)
            with self.assertRaisesRegex(ValueError, "一致しません"):
                load_checkpoint(restored, path, {**signature, "vocab_size": 33})
