import torch

from oemseg.losses.registry import build_loss


def test_registered_losses_have_finite_backward():
    for name in ("ce", "dice", "ce_dice", "ce-dice", "cedice"):
        logits = torch.randn(2, 9, 16, 16, requires_grad=True)
        target = torch.randint(0, 9, (2, 16, 16))
        loss = build_loss(name)(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
