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


@pytest.mark.parametrize("variant", ["full", "lite"])
@pytest.mark.parametrize("training", [False, True])
def test_u2net_releases_outputs_without_waiting_for_cyclic_gc(variant, training):
    import gc
    import weakref

    from oemseg.models.u2net_upstream import U2NET_full, U2NET_lite

    model = (U2NET_full if variant == "full" else U2NET_lite)(9)
    model.train(training)
    images = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 9, (2, 32, 32))
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.set_grad_enabled(training):
            outputs = model(images)
            refs = [weakref.ref(output) for output in outputs]
            if training:
                loss = U2NetDeepSupervisionLoss()(outputs, target)
                loss.backward()
                del loss
        del outputs
        assert all(ref() is None for ref in refs), (
            "U2-Net retains side/fused tensors in a recursive closure after use"
        )
    finally:
        if gc_enabled:
            gc.enable()
        gc.collect()


def test_u2net_matches_explicit_encoder_decoder_outputs_gradients_and_bn():
    import copy

    from oemseg.models.u2net_upstream import U2NET_full, _fuse_side_maps, _upsample_like

    torch.manual_seed(21)
    actual_model = U2NET_full(9)
    reference = copy.deepcopy(actual_model)
    images = torch.randn(2, 3, 33, 35)
    actual_input = images.clone().requires_grad_(True)
    reference_input = images.clone().requires_grad_(True)
    target = torch.randint(0, 9, (2, 33, 35))

    # Explicit traversal is the reference for the nested recursive forward.
    skips = []
    x = reference_input
    for height in range(1, 6):
        x = getattr(reference, f"stage{height}")(x)
        skips.append(x)
        x = reference.downsample(x)
    x = reference.stage6(x)
    sides = [_upsample_like(reference.side6(x), images.shape[-2:])]
    for height in range(5, 0, -1):
        skip = skips.pop()
        x = _upsample_like(x, skip.shape[-2:])
        x = getattr(reference, f"stage{height}d")(torch.cat((x, skip), dim=1))
        sides.append(_upsample_like(getattr(reference, f"side{height}")(x), images.shape[-2:]))
    sides.reverse()
    expected = [_fuse_side_maps(sides, reference.outconv), *sides]
    actual = actual_model(actual_input)

    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=0, atol=0)
    U2NetDeepSupervisionLoss()(actual, target).backward()
    U2NetDeepSupervisionLoss()(expected, target).backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    for (name, parameter), (other_name, other) in zip(
        actual_model.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    for name, buffer in actual_model.named_buffers():
        torch.testing.assert_close(buffer, reference.get_buffer(name), rtol=0, atol=0)
    reference.load_state_dict(actual_model.state_dict(), strict=True)
