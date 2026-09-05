"""ColabなどのCUDA環境で動かすPyTorch版のdecoder-only Transformer。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - torchなしのMac環境用
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = error
else:
    _TORCH_IMPORT_ERROR = None


TORCH_AVAILABLE = torch is not None and nn is not None and F is not None
_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> Any:
    """PyTorchを遅延要求し、未導入環境で原因を明示する。"""

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorchが必要です。Colabでは通常利用できます。ローカルでは"
            ".venv/bin/python -m pip install -e '.[torch]' を実行してください。"
        ) from _TORCH_IMPORT_ERROR
    return torch


class TorchCausalSelfAttention(_ModuleBase):
    def __init__(
        self, dim: int, heads: int, position_embedding: str = "absolute"
    ) -> None:
        require_torch()
        super().__init__()
        if dim % heads != 0:
            raise ValueError("attentionのdimはheadsで割り切れる必要があります")
        if position_embedding not in {"absolute", "rope"}:
            raise ValueError(
                "position_embedding は absolute または rope で指定してください"
            )
        self.heads = heads
        self.head_dim = dim // heads
        self.use_rope = position_embedding == "rope"
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    "RoPEではattentionのhead_dimが偶数である必要があります"
                )
            inverse = 1.0 / (
                10000
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            self.register_buffer("inverse_frequency", inverse, persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _apply_rope(self, values: Any) -> Any:
        sequence = values.shape[-2]
        positions = torch.arange(
            sequence, device=values.device, dtype=self.inverse_frequency.dtype
        )
        angles = torch.outer(positions, self.inverse_frequency)
        cosines = angles.cos()[None, None, :, :]
        sines = angles.sin()[None, None, :, :]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated = torch.empty_like(values)
        rotated[..., 0::2] = even * cosines - odd * sines
        rotated[..., 1::2] = even * sines + odd * cosines
        return rotated

    def forward(self, x: Any) -> Any:
        batch, sequence, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return self.out_proj(attended)


class TorchFeedForward(_ModuleBase):
    def __init__(self, dim: int, mlp_ratio: int, ffn_type: str = "gelu") -> None:
        require_torch()
        super().__init__()
        if ffn_type not in {"gelu", "swiglu"}:
            raise ValueError("ffn_type はgeluまたはswigluで指定してください")
        self.ffn_type = ffn_type
        hidden_dim = _ffn_hidden_dim(dim, mlp_ratio, ffn_type)
        if ffn_type == "swiglu":
            self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: Any) -> Any:
        if self.ffn_type == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x)))


def _ffn_hidden_dim(dim: int, mlp_ratio: int, ffn_type: str) -> int:
    if ffn_type == "gelu":
        return dim * mlp_ratio
    if ffn_type == "swiglu":
        return max(1, (2 * dim * mlp_ratio) // 3)
    raise ValueError("ffn_type はgeluまたはswigluで指定してください")


class TorchTransformerBlock(_ModuleBase):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int,
        position_embedding: str = "absolute",
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
    ) -> None:
        require_torch()
        super().__init__()
        if norm_type == "layernorm":
            norm = nn.LayerNorm
        elif norm_type == "rmsnorm":
            norm = nn.RMSNorm
        else:
            raise ValueError("norm_type はlayernormまたはrmsnormで指定してください")
        self.norm_1 = norm(dim)
        self.attention = TorchCausalSelfAttention(dim, heads, position_embedding)
        self.norm_2 = norm(dim)
        self.mlp = TorchFeedForward(dim, mlp_ratio, ffn_type)

    def forward(self, x: Any) -> Any:
        x = x + self.attention(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class TorchJapaneseGPT(_ModuleBase):
    """MLX版と同じ構造を持つ、Colab用の重み共有GPT。"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        context_length: int,
        mlp_ratio: int = 4,
        position_embedding: str = "absolute",
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
    ) -> None:
        require_torch()
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size は正の整数で指定してください")
        if dim <= 0 or layers <= 0 or heads <= 0 or context_length < 2:
            raise ValueError(
                "dim、layers、headsは正数、context_lengthは2以上で指定してください"
            )
        if dim % heads != 0:
            raise ValueError("dimはheadsで割り切れる必要があります")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratioは正の整数で指定してください")
        if position_embedding not in {"absolute", "rope"}:
            raise ValueError(
                "position_embedding は absolute または rope で指定してください"
            )
        if norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type はlayernormまたはrmsnormで指定してください")
        if ffn_type not in {"gelu", "swiglu"}:
            raise ValueError("ffn_type はgeluまたはswigluで指定してください")
        if position_embedding == "rope" and (dim // heads) % 2 != 0:
            raise ValueError("RoPEではattentionのhead_dimが偶数である必要があります")
        self.vocab_size = vocab_size
        self.dim = dim
        self.layers_count = layers
        self.heads = heads
        self.context_length = context_length
        self.mlp_ratio = mlp_ratio
        self.position_embedding_type = position_embedding
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.token_embedding = nn.Embedding(vocab_size, dim)
        if position_embedding == "absolute":
            self.position_embedding = nn.Embedding(context_length, dim)
        self.blocks = nn.ModuleList(
            [
                TorchTransformerBlock(
                    dim, heads, mlp_ratio, position_embedding, norm_type, ffn_type
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = (
            nn.LayerNorm(dim) if norm_type == "layernorm" else nn.RMSNorm(dim)
        )
        # PyTorchのEmbedding既定値は標準偏差1の正規分布ですが、MLXの
        # 小型モデル実験で使っている初期スケールとは大きく異なります。
        # 入力とabsolute positionを同じ小さなスケールへ揃え、weight tying
        # された出力logitsが初期状態で過大にならないようにします。
        with torch.no_grad():
            embedding_std = 1.0 / math.sqrt(dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=embedding_std)
            if position_embedding == "absolute":
                nn.init.normal_(
                    self.position_embedding.weight,
                    mean=0.0,
                    std=embedding_std,
                )

    def forward(self, tokens: Any) -> Any:
        if tokens.ndim != 2:
            raise ValueError(
                f"tokensは[batch, sequence]が必要ですが、shape={tuple(tokens.shape)}です"
            )
        _, sequence = tokens.shape
        if sequence > self.context_length:
            raise ValueError(
                f"入力長{sequence}がcontext_length={self.context_length}を超えています"
            )
        x = self.token_embedding(tokens)
        if self.position_embedding_type == "absolute":
            positions = torch.arange(sequence, device=tokens.device)
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.token_embedding.weight.T


def load_mlx_weights(model: Any, path: str | Path) -> None:
    """MLXのnpz checkpointを、同名のPyTorch state_dictへ読み込む。"""

    require_torch()
    weights = np.load(Path(path))
    expected = model.state_dict()
    actual_keys = set(weights.files)
    expected_keys = set(expected)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            "MLX checkpointのキーが一致しません。"
            f" missing={sorted(missing)}, extra={sorted(extra)}"
        )
    converted = {
        key: torch.from_numpy(np.asarray(weights[key])).to(
            device=expected[key].device, dtype=expected[key].dtype
        )
        for key in expected
    }
    model.load_state_dict(converted, strict=True)


def parameter_count(model: Any) -> int:
    """重み共有を反映した実際の学習parameter数を返す。"""

    require_torch()
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
