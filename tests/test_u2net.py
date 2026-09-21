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


def test_u2net_fusion_matches_concat_forward_and_gradients_without_cat(monkeypatch):
    import copy

    import oemseg.models.u2net_upstream as upstream

    assert hasattr(upstream, "_fuse_side_maps"), "memory-efficient U2-Net fusion helper is missing"
    fuse = upstream._fuse_side_maps

    torch.manual_seed(7)
    channels = 3
    original_maps = [torch.randn(2, channels, 8, 8, requires_grad=True) for _ in range(6)]
    optimized_maps = [tensor.detach().clone().requires_grad_(True) for tensor in original_maps]
    original_conv = torch.nn.Conv2d(6 * channels, channels, kernel_size=1)
    optimized_conv = copy.deepcopy(original_conv)

    expected = original_conv(torch.cat(original_maps, dim=1))
    expected.square().mean().backward()

    def forbid_cat(*args, **kwargs):
        raise AssertionError("U2-Net fused head must not materialize the 6-way concatenation")

    monkeypatch.setattr(torch, "cat", forbid_cat)
    actual = fuse(optimized_maps, optimized_conv)
    actual.square().mean().backward()

    assert torch.allclose(actual, expected.detach(), rtol=1e-5, atol=1e-6)
    for actual_map, expected_map in zip(optimized_maps, original_maps):
        assert torch.allclose(actual_map.grad, expected_map.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(optimized_conv.weight.grad, original_conv.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(optimized_conv.bias.grad, original_conv.bias.grad, rtol=1e-5, atol=1e-6)
