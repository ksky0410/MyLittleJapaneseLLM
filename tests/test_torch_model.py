import unittest

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from my_little_japanese_llm.torch_model import (
    TorchJapaneseGPT,
    parameter_count,
)


HAS_TORCH = torch is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch未導入")
class TorchModelTests(unittest.TestCase):
    def test_forward_shape_and_weight_tying(self) -> None:
        model = TorchJapaneseGPT(32, 16, 2, 4, 8, 2)
        logits = model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertEqual(tuple(logits.shape), (2, 4, 32))
        self.assertGreater(parameter_count(model), 0)

    def test_rope_forward_shape(self) -> None:
        model = TorchJapaneseGPT(32, 16, 2, 4, 8, 2, "rope")
        logits = model(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(tuple(logits.shape), (1, 4, 32))
