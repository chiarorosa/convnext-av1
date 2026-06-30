"""ConvNeXt-AV1 model."""

from __future__ import annotations


class AV1ConvAdapter:
    def __new__(cls, channels: int, bottleneck_ratio: float = 0.25):
        import torch
        import torch.nn as nn

        class _Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                bottleneck = max(1, int(channels * bottleneck_ratio))
                self.adapter = nn.Sequential(
                    nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
                    nn.BatchNorm2d(bottleneck),
                    nn.GELU(),
                    nn.Conv2d(bottleneck, bottleneck, kernel_size=7, padding=3, groups=bottleneck, bias=False),
                    nn.BatchNorm2d(bottleneck),
                    nn.GELU(),
                    nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                self.s = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x + self.s * self.adapter(x)

        return _Adapter()


class QPConditionedHead:
    def __new__(cls, in_channels: int, num_classes: int = 10, hidden_dim: int = 128):
        import torch
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Sequential(
                    nn.Linear(in_channels + 1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x, qp):
                if qp.ndim == 1:
                    qp_view = qp.view(-1, 1)
                else:
                    qp_view = qp
                pooled = self.pool(x).flatten(1)
                return self.classifier(torch.cat([pooled, qp_view], dim=1))

        return _Head()


def _convnext_tiny_features(pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    base_model = models.convnext_tiny(weights=weights)
    old_stem = base_model.features[0][0]
    new_stem = nn.Conv2d(1, 96, kernel_size=4, stride=4)
    if pretrained:
        with __import__("torch").no_grad():
            new_stem.weight.copy_(old_stem.weight.mean(dim=1, keepdim=True))
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)
    base_model.features[0][0] = new_stem
    return base_model


class ConvNeXtAV1:
    def __new__(cls, num_av1_classes: int = 10, pretrained: bool = False):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                base_model = _convnext_tiny_features(pretrained=pretrained)
                for param in base_model.parameters():
                    param.requires_grad = False
                for param in base_model.features[0][0].parameters():
                    param.requires_grad = True

                self.stage1 = base_model.features[0:2]
                self.stage2 = base_model.features[2:4]
                self.stage3 = base_model.features[4:6]
                self.adapter1 = AV1ConvAdapter(96)
                self.adapter2 = AV1ConvAdapter(192)
                self.adapter3 = AV1ConvAdapter(384)
                self.head_64x64 = QPConditionedHead(96, num_av1_classes)
                self.head_32x32 = QPConditionedHead(192, num_av1_classes)
                self.head_16x16 = QPConditionedHead(384, num_av1_classes)

            def forward(self, x, qp):
                x1 = self.adapter1(self.stage1(x))
                logits_64 = self.head_64x64(x1, qp)
                x2 = self.adapter2(self.stage2(x1))
                logits_32 = self.head_32x32(x2, qp)
                x3 = self.adapter3(self.stage3(x2))
                logits_16 = self.head_16x16(x3, qp)
                return {
                    "partition_64x64": logits_64,
                    "partition_32x32": logits_32,
                    "partition_16x16": logits_16,
                }

        return _Model()
