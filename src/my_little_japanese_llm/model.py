"""MLXで動く最小のdecoder-only Transformer。"""

from __future__ import annotations

import math
from typing import Any

try:
    import mlx.core as mx
    from mlx import nn
except ImportError as error:  # pragma: no cover - MLXなし環境用の分岐
    mx = None
    nn = None
    _MLX_IMPORT_ERROR = error
else:
    _MLX_IMPORT_ERROR = None

MLX_AVAILABLE = mx is not None and nn is not None


def require_mlx() -> Any:
    """MLXを遅延要求し、学習系だけを明確なエラーで止める。"""

    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLXまたはMetalデバイスを利用できません。この学習コードは"
            "Apple Silicon Mac向けです。Metalが使える環境で、"
            ".venv/bin/python -m pip install -e '.[apple,dev]' を実行してください。"
        ) from _MLX_IMPORT_ERROR
    return mx


_ModuleBase = nn.Module if nn is not None else object


class CausalSelfAttention(_ModuleBase):
    def __init__(
        self, dim: int, heads: int, position_embedding: str = "absolute"
    ) -> None:
        require_mlx()
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
            self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: Any) -> Any:
        batch, sequence, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(batch, sequence, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, sequence, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, sequence, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # 上三角だけを十分小さい値にし、未来tokenを参照できないようにする。
        mask = mx.triu(mx.full((sequence, sequence), -1e9, dtype=scores.dtype), k=1)
        weights = mx.softmax(scores + mask, axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, sequence, -1)
        return self.out_proj(attended)


class FeedForward(_ModuleBase):
    def __init__(self, dim: int, mlp_ratio: int, ffn_type: str = "gelu") -> None:
        require_mlx()
        super().__init__()
        if ffn_type not in {"gelu", "swiglu"}:
            raise ValueError("ffn_type はgeluまたはswigluで指定してください")
        self.ffn_type = ffn_type
        hidden_dim = _ffn_hidden_dim(dim, mlp_ratio, ffn_type)
        if ffn_type == "swiglu":
            self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def __call__(self, x: Any) -> Any:
        if self.ffn_type == "swiglu":
            gate = self.gate(x)
            return self.down(mx.sigmoid(gate) * gate * self.up(x))
        return self.down(nn.gelu(self.up(x)))


def _ffn_hidden_dim(dim: int, mlp_ratio: int, ffn_type: str) -> int:
    if ffn_type == "gelu":
        return dim * mlp_ratio
    if ffn_type == "swiglu":
        # SwiGLUはgate/up/downの3射影を使うため、GELUの2射影と
        # 近いparameter予算になるよう2/3倍の中間次元にする。
        return max(1, (2 * dim * mlp_ratio) // 3)
    raise ValueError("ffn_type はgeluまたはswigluで指定してください")


def _make_norm(dim: int, norm_type: str) -> Any:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError("norm_type はlayernormまたはrmsnormで指定してください")


class TransformerBlock(_ModuleBase):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int,
        position_embedding: str = "absolute",
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
    ) -> None:
        require_mlx()
        super().__init__()
        self.norm_1 = _make_norm(dim, norm_type)
        self.attention = CausalSelfAttention(dim, heads, position_embedding)
        self.norm_2 = _make_norm(dim, norm_type)
        self.mlp = FeedForward(dim, mlp_ratio, ffn_type)

    def __call__(self, x: Any) -> Any:
        x = x + self.attention(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class TinyJapaneseGPT(_ModuleBase):
    """学習教材として読みやすさを優先した、重み共有付きの小型GPT。"""

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
        require_mlx()
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
            raise ValueError(
                "norm_type はlayernormまたはrmsnormで指定してください"
            )
        if ffn_type not in {"gelu", "swiglu"}:
            raise ValueError(
                "ffn_type はgeluまたはswigluで指定してください"
            )
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
        self.blocks = [
            TransformerBlock(
                dim, heads, mlp_ratio, position_embedding, norm_type, ffn_type
            )
            for _ in range(layers)
        ]
        self.final_norm = _make_norm(dim, norm_type)

    def __call__(self, tokens: Any) -> Any:
        if len(tokens.shape) != 2:
            raise ValueError(
                f"tokensは[batch, sequence]が必要ですが、shape={tokens.shape}です"
            )
        _, sequence = tokens.shape
        if sequence > self.context_length:
            raise ValueError(
                f"入力長{sequence}がcontext_length={self.context_length}を超えています"
            )
        x = self.token_embedding(tokens)
        if self.position_embedding_type == "absolute":
            positions = mx.arange(sequence)
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # lm_headを別に持たず、入力Embeddingのweightを転置して出力にも使う。
        return x @ self.token_embedding.weight.T


def model_signature(
    vocab_size: int,
    dim: int,
    layers: int,
    heads: int,
    context_length: int,
    mlp_ratio: int,
    position_embedding: str = "absolute",
    norm_type: str = "layernorm",
    ffn_type: str = "gelu",
) -> dict[str, int | str]:
    if position_embedding not in {"absolute", "rope"}:
        raise ValueError(
            "position_embedding は absolute または rope で指定してください"
        )
    if position_embedding == "rope" and (dim // heads) % 2 != 0:
        raise ValueError("RoPEではattentionのhead_dimが偶数である必要があります")
    if norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError("norm_type はlayernormまたはrmsnormで指定してください")
    if ffn_type not in {"gelu", "swiglu"}:
        raise ValueError("ffn_type はgeluまたはswigluで指定してください")
    return {
        "vocab_size": int(vocab_size),
        "dim": int(dim),
        "layers": int(layers),
        "heads": int(heads),
        "context_length": int(context_length),
        "mlp_ratio": int(mlp_ratio),
        "position_embedding": position_embedding,
        "norm_type": norm_type,
        "ffn_type": ffn_type,
    }


def estimate_parameter_count(
    vocab_size: int,
    dim: int,
    layers: int,
    heads: int,
    context_length: int,
    mlp_ratio: int = 4,
    position_embedding: str = "absolute",
    norm_type: str = "layernorm",
    ffn_type: str = "gelu",
) -> int:
    """重み共有を反映した概算。MLXなしのinspectでも利用できる。"""

    if dim % heads != 0:
        raise ValueError("dimはheadsで割り切れる必要があります")
    if position_embedding not in {"absolute", "rope"}:
        raise ValueError(
            "position_embedding は absolute または rope で指定してください"
        )
    if position_embedding == "rope" and (dim // heads) % 2 != 0:
        raise ValueError("RoPEではattentionのhead_dimが偶数である必要があります")
    if norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError("norm_type はlayernormまたはrmsnormで指定してください")
    if ffn_type not in {"gelu", "swiglu"}:
        raise ValueError("ffn_type はgeluまたはswigluで指定してください")
    token_embedding = vocab_size * dim
    position_embedding_parameters = (
        context_length * dim if position_embedding == "absolute" else 0
    )
    attention = dim * dim * 3 + dim * dim
    hidden_dim = _ffn_hidden_dim(dim, mlp_ratio, ffn_type)
    projection_count = 3 if ffn_type == "swiglu" else 2
    mlp = projection_count * dim * hidden_dim
    block_norms = 2 * dim if norm_type == "rmsnorm" else 4 * dim
    final_norm = dim if norm_type == "rmsnorm" else 2 * dim
    return (
        token_embedding
        + position_embedding_parameters
        + layers * (attention + mlp + block_norms)
        + final_norm
    )
