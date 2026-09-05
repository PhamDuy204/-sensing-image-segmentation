import torch
from torch import nn

from oemseg.losses.mask2former import build_mask2former_targets
from oemseg.losses import repstdc as repstdc_losses


def test_mask2former_dense_targets_preserve_present_classes():
    targets = torch.tensor([[[0, 0, 2], [2, 5, 5]]])
    masks, labels = build_mask2former_targets(targets)

    assert labels[0].tolist() == [0, 2, 5]
    assert masks[0].shape == (3, 2, 3)
    assert torch.equal(masks[0][1].bool(), targets[0].eq(2))


def test_repstdc_native_loss_sums_main_and_all_auxiliary_terms(monkeypatch):
    monkeypatch.setattr(repstdc_losses, "_data_samples", lambda targets: [object()] * len(targets))

    class FakeHead(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def loss(self, features, samples, train_cfg):
            del samples, train_cfg
            return {
                "loss_ce": features.mean() * self.scale,
                "acc_seg": features.new_tensor(0.5),
            }

    features = torch.ones(1, 1, 2, 2, requires_grad=True)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    loss = repstdc_losses.repstdc_native_loss(
        FakeHead(1.0),
        [FakeHead(2.0), FakeHead(3.0), FakeHead(4.0)],
        features,
        targets,
    )

    assert torch.isclose(loss, torch.tensor(10.0))
    loss.backward()
    assert features.grad is not None


def test_mask2former_reporting_loss_accepts_positive_semantic_scores():
    from oemseg.losses.mask2former import build_mask2former_reporting_loss

    scores = torch.rand(2, 9, 8, 8)
    target = torch.randint(0, 9, (2, 8, 8))
    loss = build_mask2former_reporting_loss()(scores, target)
    assert torch.isfinite(loss)
