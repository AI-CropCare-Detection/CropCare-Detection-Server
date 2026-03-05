"""
tta.py  –  Test-Time Augmentation transforms.
Extracted exactly from the training notebook (cell 5).
IMG_SIZE and NORM are read from config so they stay in sync.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, NORM

_SZ = CFG["IMG_SIZE"]

# Three deterministic views used during TTA inference:
#   1. clean centre crop
#   2. horizontal flip
#   3. vertical flip
tta_transforms: list[A.Compose] = [
    A.Compose([A.Resize(_SZ, _SZ),                      A.Normalize(**NORM), ToTensorV2()]),
    A.Compose([A.Resize(_SZ, _SZ), A.HorizontalFlip(p=1), A.Normalize(**NORM), ToTensorV2()]),
    A.Compose([A.Resize(_SZ, _SZ), A.VerticalFlip(p=1),   A.Normalize(**NORM), ToTensorV2()]),
]
