import torch

from oemseg.models.registry import build_model_from_values


def assert_adapter_contract(model):
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_unetpp_adapter_contract():
    model = build_model_from_values("unetpp", "resnet18", pretrained=False)
    assert_adapter_contract(model)


def test_unet_adapter_contract():
    model = build_model_from_values("unet", "resnet18", pretrained=False)
    assert_adapter_contract(model)


def test_segformer_adapter_contract_without_download():
    model = build_model_from_values("segformer", "b0", pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    logits.mean().backward()
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_upernet_decoder_contract():
    from oemseg.models.upernet import UPerNetHead

    decoder = UPerNetHead((80, 160, 320, 640), channels=64, num_classes=9)
    features = [
        torch.randn(2, 80, 32, 32, requires_grad=True),
        torch.randn(2, 160, 16, 16, requires_grad=True),
        torch.randn(2, 320, 8, 8, requires_grad=True),
        torch.randn(2, 640, 4, 4, requires_grad=True),
    ]
    logits = decoder(features)
    assert logits.shape == (2, 9, 32, 32)
    logits.mean().backward()


def test_mambavision_adapter_contract_with_fake_backbone():
    from torch import nn
    from oemseg.models.mambavision import MambaVisionAdapter

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([
                nn.Conv2d(3, 80, 4, 4),
                nn.Conv2d(80, 160, 2, 2),
                nn.Conv2d(160, 320, 2, 2),
                nn.Conv2d(320, 640, 2, 2),
            ])
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(640)
            self.model.head = nn.Linear(640, 1000)

        def forward(self, images):
            features = []
            x = images
            for stage in self.stages:
                x = stage(x)
                features.append(x)
            return self.model.norm(x.mean((2, 3))), features

    model = MambaVisionAdapter(
        variant="tiny",
        pretrained=False,
        backbone=FakeBackbone(),
        decoder_channels=64,
    )
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])
