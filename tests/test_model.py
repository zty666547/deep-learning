import pytest

torch = pytest.importorskip("torch")

from microc_foundation.model import MicroCCNN


def test_model_preserves_batch_and_emits_class_logits():
    model = MicroCCNN(input_channels=2, num_classes=3)
    result = model(torch.zeros(4, 2, 128, 128))
    assert result.shape == (4, 3)
