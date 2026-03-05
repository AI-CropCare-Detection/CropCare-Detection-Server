from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building blocks ─────────────────────────────────────────────────────

class CBS(nn.Module):
    """Conv → BatchNorm → SiLU"""
    def __init__(self, c_in, c_out, k=1, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = CBS(c, c, 1)
        self.cv2 = CBS(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        c_ = c_out // 2
        self.cv1 = CBS(c_in, c_, 1)
        self.cv2 = nn.Conv2d(c_in, c_, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, bias=False)
        self.cv4 = CBS(2 * c_, c_out, 1)
        self.bn  = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU(inplace=True)
        self.m   = nn.Sequential(*[Bottleneck(c_) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat([y1, y2], dim=1))))


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block (YOLOv7)."""
    def __init__(self, c_in, c_out, c_mid, n=4):
        super().__init__()
        self.cv1 = CBS(c_in,  c_mid, 1)
        self.cv2 = CBS(c_in,  c_mid, 1)
        self.cvs = nn.ModuleList([CBS(c_mid, c_mid, 3) for _ in range(n)])
        self.cv7 = CBS(c_mid * (n + 2), c_out, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        xs = [x1, x2]
        xi = x2
        for cv in self.cvs:
            xi = cv(xi)
            xs.append(xi)
        return self.cv7(torch.cat(xs, dim=1))


class SPPCSPC(nn.Module):
    """Spatial Pyramid Pooling with CSP."""
    def __init__(self, c_in, c_out, k=(5, 9, 13)):
        super().__init__()
        c_ = c_out
        self.cv1 = CBS(c_in,  c_,  1)
        self.cv2 = CBS(c_in,  c_,  1)
        self.cv3 = CBS(c_,    c_,  3)
        self.cv4 = CBS(c_,    c_,  1)
        self.m   = nn.ModuleList([nn.MaxPool2d(ki, 1, ki // 2) for ki in k])
        self.cv5 = CBS(4 * c_, c_, 1)
        self.cv6 = CBS(c_,     c_, 3)
        self.cv7 = CBS(2 * c_, c_out, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1, *[m(x1) for m in self.m]], dim=1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat([y1, y2], dim=1))


# ── Backbone ──────────────────────────────────────────────────────────────────

class YOLOv7Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(CBS(3, 32, 3, 1), CBS(32, 64, 3, 2), CBS(64, 64, 3, 1))
        self.stage1 = nn.Sequential(CBS(64, 128, 3, 2), BottleneckCSP(128, 128, n=3))
        self.stage2 = nn.Sequential(CBS(128, 256, 3, 2), ELAN(256, 256, 128, n=4))
        self.stage3 = nn.Sequential(CBS(256, 512, 3, 2), ELAN(512, 512, 256, n=4))
        self.stage4 = nn.Sequential(
            CBS(512, 1024, 3, 2), ELAN(1024, 1024, 512, n=4), SPPCSPC(1024, 512)
        )
        self.out_channels = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# ── Classifier ────────────────────────────────────────────────────────────────

class YOLOv7Classifier(nn.Module):
    """
    YOLOv7 Backbone + Attention Pooling + Deep MLP head.

    Input  → (B, 3, H, W)
    Output → (B, num_classes)  raw logits
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = YOLOv7Backbone()
        fd = self.backbone.out_channels  # 512

        # Spatial attention to focus on disease-relevant regions
        self.attn = nn.Sequential(
            nn.Conv2d(fd, fd // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(fd // 8, 1, 1),  nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fd * 2, 1024), nn.BatchNorm1d(1024), nn.SiLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, 512),    nn.BatchNorm1d(512),  nn.SiLU(inplace=True), nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f   = self.backbone(x)                              # (B, 512, h, w)
        f   = f * self.attn(f)                              # spatial attention
        out = torch.cat([self.gap(f), self.gmp(f)], dim=1) # (B, 1024, 1, 1)
        return self.head(out)
