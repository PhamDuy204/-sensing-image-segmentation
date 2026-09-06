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


def test_unetformer_adapter_contract_with_fake_upstream():
    from torch import nn
    from oemseg.models.unetformer import UNetFormerAdapter

    class FakeUNetFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 8, 3, padding=1)
            self.decoder = nn.Module()
            self.decoder.head = nn.Conv2d(8, 9, 1)
            self.decoder.aux_head = nn.Conv2d(8, 9, 1)

        def forward(self, images):
            features = self.backbone(images)
            logits = self.decoder.head(features)
            if self.training:
                return logits, self.decoder.aux_head(features)
            return logits

    model = UNetFormerAdapter(model=FakeUNetFormer())
    model.train()
    logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 9, 64, 64)
    assert all(parameter.requires_grad for parameter in model.model.decoder.aux_head.parameters())
    logits.mean().backward()
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_unetformer_uses_official_auxiliary_loss_when_targets_are_given():
    from oemseg.models.unetformer import UNetFormerAdapter

    model = UNetFormerAdapter(variant="resnet18", pretrained=False, decoder_channels=64)
    model.train()
    images = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 9, (2, 64, 64))

    loss = model(images, targets=targets)
    assert model.backbone_name == "swsl_resnet18"
    assert model.uses_native_loss is True
    assert loss.ndim == 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.model.decoder.aux_head.parameters())


def test_unetformer_swin_b_uses_official_ft_factory(monkeypatch):
    from types import SimpleNamespace
    from torch import nn
    from oemseg.models import unetformer as module

    captured = {}

    class FakeFTUNetFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 8, 3, padding=1)
            self.decoder = nn.Conv2d(8, 9, 1)

        def forward(self, images):
            return self.decoder(self.backbone(images))

    def build_ft(**kwargs):
        captured.update(kwargs)
        return FakeFTUNetFormer()

    monkeypatch.setattr(
        module,
        "_load_ft_upstream",
        lambda: SimpleNamespace(ft_unetformer=build_ft),
        raising=False,
    )
    module.sys.path.insert(0, str(module.GEOSEG_DIR))
    model = module.UNetFormerAdapter(variant="swin-b", pretrained=False, decoder_channels=256)

    assert captured["pretrained"] is False
    assert captured["num_classes"] == 9
    assert captured["decoder_channels"] == 256
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 9, 64, 64)
    assert model.uses_native_loss is True


def test_pyramidmamba_adapter_contract_with_fake_upstream():
    from torch import nn
    from oemseg.models.pyramidmamba import PyramidMambaAdapter

    class FakePyramidMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 8, 3, padding=1)
            self.decoder = nn.Conv2d(8, 9, 1)

        def forward(self, images):
            return self.decoder(self.backbone(images))[:, :, ::2, ::2]

    model = PyramidMambaAdapter(model=FakePyramidMamba())
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    logits.mean().backward()
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_segnext_adapter_contract_with_fake_openmmlab_components():
    from torch import nn
    from oemseg.models.segnext import SegNeXtAdapter

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, images):
            feature = self.conv(images)
            return (feature, feature, feature, feature)

    class FakeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Conv2d(8, 9, 1)

        def forward(self, features):
            return self.classifier(features[-1])[:, :, ::2, ::2]

    model = SegNeXtAdapter(pretrained=False, backbone=FakeBackbone(), decode_head=FakeHead())
    assert_adapter_contract(model)


def test_repstdc_adapter_contract_with_fake_official_components():
    from torch import nn
    from oemseg.models.repstdc import RepSTDCAdapter

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, images):
            feature = self.conv(images)
            return (feature, feature, feature, feature)

    class FakeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Conv2d(8, 9, 1)

        def forward(self, features):
            return self.classifier(features[-1])[:, :, ::2, ::2]

    model = RepSTDCAdapter(pretrained=False, backbone=FakeBackbone(), decode_head=FakeHead())
    assert_adapter_contract(model)


def test_mask2former_adapter_dense_logits_and_native_loss_with_fake_model():
    from types import SimpleNamespace
    from torch import nn
    from oemseg.models.mask2former import Mask2FormerAdapter

    class FakeMask2Former(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(3, 4, 1)
            self.mask_head = nn.Conv2d(4, 3, 1)
            self.class_queries = nn.Parameter(torch.randn(3, 10))

        def forward(self, pixel_values, mask_labels=None, class_labels=None):
            features = self.encoder(pixel_values)
            masks = self.mask_head(features)[:, :, ::2, ::2]
            classes = self.class_queries.unsqueeze(0).expand(pixel_values.shape[0], -1, -1)
            loss = None
            if mask_labels is not None and class_labels is not None:
                loss = masks.mean() + classes.mean()
            return SimpleNamespace(
                class_queries_logits=classes,
                masks_queries_logits=masks,
                loss=loss,
            )

    fake = FakeMask2Former()
    model = Mask2FormerAdapter(pretrained=False, model=fake, backbone=fake.encoder)
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 9, 64, 64)
    target = torch.randint(0, 9, (1, 64, 64))
    loss = model(x, targets=target)
    assert loss.ndim == 0
    loss.backward()
    groups = model.parameter_groups(base_lr=6e-4, backbone_lr=6e-5)
    ids = [{id(p) for p in group["params"]} for group in groups]
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


def test_segnext_decode_head_runs_in_float32_under_autocast():
    from torch import nn
    from oemseg.models.segnext import SegNeXtAdapter

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 1)
            self.output_dtype = None

        def forward(self, images):
            feature = self.conv(images)
            self.output_dtype = feature.dtype
            return (feature, feature, feature, feature)

    class FakeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Conv2d(8, 9, 1)
            self.input_dtype = None

        def forward(self, features):
            self.input_dtype = features[-1].dtype
            return self.classifier(features[-1])

    backbone = FakeBackbone()
    head = FakeHead()
    model = SegNeXtAdapter(pretrained=False, backbone=backbone, decode_head=head)
    x = torch.randn(1, 3, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
    assert backbone.output_dtype == torch.float32
    assert head.input_dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_pyramidmamba_repairs_zeroed_dt_projection_and_runs_mamba_block_in_float32():
    from torch import nn
    from oemseg.models.pyramidmamba import PyramidMambaAdapter

    class FakeMambaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dt_rank = 2
            self.d_inner = 8
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.dt_proj.bias._no_reinit = True
            nn.init.zeros_(self.dt_proj.weight)
            nn.init.zeros_(self.dt_proj.bias)
            self.proj = nn.Conv2d(8, 8, 1)
            self.input_dtype = None

        def forward(self, x):
            self.input_dtype = x.dtype
            return self.proj(x)

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = FakeMambaLayer()

        def forward(self, x):
            return self.mamba(x)

    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.b3 = FakeBlock()
            self.head = nn.Conv2d(8, 9, 1)

        def forward(self, x):
            return self.head(self.b3(x))

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 1)
            self.output_dtype = None

        def forward(self, images):
            features = self.conv(images)
            self.output_dtype = features.dtype
            return features

    class FakePyramidMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.decoder = FakeDecoder()

        def forward(self, images):
            return self.decoder(self.backbone(images))

    upstream = FakePyramidMamba()
    mamba = upstream.decoder.b3.mamba
    model = PyramidMambaAdapter(model=upstream)

    assert not torch.equal(mamba.dt_proj.bias, torch.zeros_like(mamba.dt_proj.bias))
    assert getattr(mamba.dt_proj.bias, "_no_weight_decay", False)

    x = torch.randn(1, 3, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
    assert upstream.backbone.output_dtype == torch.float32
    assert mamba.input_dtype == torch.float32
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_mask2former_marks_official_norm_embedding_and_position_parameters_no_decay():
    from torch import nn
    from oemseg.models.mask2former import Mask2FormerAdapter

    class FakeMask2Former(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GroupNorm(1, 4))
            self.norm = nn.LayerNorm(4)
            self.embedding = nn.Embedding(4, 4)
            self.relative_position_bias_table = nn.Parameter(torch.randn(4, 4))

    fake = FakeMask2Former()
    Mask2FormerAdapter(pretrained=False, model=fake, backbone=fake.encoder)

    assert not getattr(fake.encoder[0].weight, "_no_weight_decay", False)
    assert getattr(fake.encoder[1].weight, "_no_weight_decay", False)
    assert getattr(fake.encoder[1].bias, "_no_weight_decay", False)
    assert getattr(fake.norm.weight, "_no_weight_decay", False)
    assert getattr(fake.embedding.weight, "_no_weight_decay", False)
    assert getattr(fake.relative_position_bias_table, "_no_weight_decay", False)
