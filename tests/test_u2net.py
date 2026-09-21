import pytest
import torch
import torch.nn.functional as F

from oemseg.losses.registry import resolve_loss_name
from oemseg.losses.u2net import U2NetDeepSupervisionLoss
from oemseg.models.registry import build_model_from_values


def test_u2net_lite_adapter_contract_and_native_loss():
    model = build_model_from_values("u2net", "lite", pretrained=False)
    images = torch.randn(1, 3, 64, 64)
    targets = torch.randint(0, 9, (1, 64, 64))

    logits = model(images)
    assert logits.shape == (1, 9, 64, 64)

    loss = model(images, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])
    assert [group["lr"] for group in groups] == [6e-4, 6e-4]


def test_u2net_deep_supervision_is_equal_weight_sum_of_seven_ce_terms():
    outputs = [torch.randn(2, 9, 8, 8, requires_grad=True) for _ in range(7)]
    target = torch.randint(0, 9, (2, 8, 8))
    expected = sum(F.cross_entropy(output, target) for output in outputs)

    loss = U2NetDeepSupervisionLoss()(outputs, target)

    assert torch.allclose(loss, expected)
    loss.backward()
    assert all(output.grad is not None for output in outputs)


def test_u2net_auto_loss_is_native_and_dense_override_is_rejected():
    assert resolve_loss_name("auto", "u2net", "full") == "u2net"
    with pytest.raises(ValueError, match="requires --loss u2net"):
        resolve_loss_name("ce_dice", "u2net", "full")
