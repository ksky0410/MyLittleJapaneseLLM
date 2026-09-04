"""TOML設定を読み込み、実験で使う型付き設定を提供する。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    prepared_dir: Path
    tokenizer_model: Path
    train_tokens: Path
    val_tokens: Path
    checkpoint_dir: Path
    samples_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    layers: int
    heads: int
    context_length: int
    mlp_ratio: int
    position_embedding: str

    def validate(self) -> None:
        if self.dim <= 0 or self.layers <= 0 or self.heads <= 0:
            raise ValueError(
                "model.dim, model.layers, model.heads は正の整数で指定してください"
            )
        if self.dim % self.heads != 0:
            raise ValueError("model.dim は model.heads で割り切れる必要があります")
        if self.context_length < 2:
            raise ValueError("model.context_length は2以上で指定してください")
        if self.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio は正の整数で指定してください")
        if self.position_embedding not in {"absolute", "rope"}:
            raise ValueError(
                "model.position_embedding はabsoluteまたはropeで指定してください"
            )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_steps: int
    eval_interval: int
    sample_interval: int
    eval_batches: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    seed: int

    def validate(self) -> None:
        fields = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "sample_interval": self.sample_interval,
            "eval_batches": self.eval_batches,
            "warmup_steps": self.warmup_steps,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"training.{name} は正の整数で指定してください")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("学習率は正数、最小学習率は0以上で指定してください")


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens は正の整数で指定してください")
        if self.temperature <= 0:
            raise ValueError("generation.temperature は正数で指定してください")
        if self.top_k < 0:
            raise ValueError("generation.top_k は0以上で指定してください")


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    source_path: Path

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.generation.validate()


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> ExperimentConfig:
    """TOMLを読み込む。相対パスはリポジトリルートから解決する。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {source}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    root = source.parent.parent
    paths_raw = raw["paths"]
    model_raw = raw["model"]
    training_raw = raw["training"]
    generation_raw = raw["generation"]
    config = ExperimentConfig(
        paths=PathsConfig(
            prepared_dir=_path(root, paths_raw["prepared_dir"]),
            tokenizer_model=_path(root, paths_raw["tokenizer_model"]),
            train_tokens=_path(root, paths_raw["train_tokens"]),
            val_tokens=_path(root, paths_raw["val_tokens"]),
            checkpoint_dir=_path(root, paths_raw["checkpoint_dir"]),
            samples_dir=_path(root, paths_raw["samples_dir"]),
        ),
        model=ModelConfig(
            dim=int(model_raw["dim"]),
            layers=int(model_raw["layers"]),
            heads=int(model_raw["heads"]),
            context_length=int(model_raw["context_length"]),
            mlp_ratio=int(model_raw.get("mlp_ratio", 4)),
            position_embedding=str(model_raw.get("position_embedding", "absolute")),
        ),
        training=TrainingConfig(
            batch_size=int(training_raw["batch_size"]),
            max_steps=int(training_raw["max_steps"]),
            eval_interval=int(training_raw["eval_interval"]),
            sample_interval=int(training_raw["sample_interval"]),
            eval_batches=int(training_raw["eval_batches"]),
            learning_rate=float(training_raw["learning_rate"]),
            min_learning_rate=float(training_raw["min_learning_rate"]),
            warmup_steps=int(training_raw["warmup_steps"]),
            weight_decay=float(training_raw["weight_decay"]),
            seed=int(training_raw["seed"]),
        ),
        generation=GenerationConfig(
            prompt=str(generation_raw["prompt"]),
            max_new_tokens=int(generation_raw["max_new_tokens"]),
            temperature=float(generation_raw["temperature"]),
            top_k=int(generation_raw["top_k"]),
        ),
        source_path=source,
    )
    config.validate()
    return config
