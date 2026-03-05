"""
config.py  –  Central configuration (values match the training notebook exactly).
"""

from __future__ import annotations

import torch

# ── Hardware ──────────────────────────────────────────────────────────────────

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Normalisation — ImageNet stats used during training ───────────────────────
# NOTE: notebook uses std=(0.229, 0.224, 0.225) — must match training exactly

NORM: dict = {
    "mean": (0.485, 0.456, 0.406),
    "std" : (0.229, 0.224, 0.225),
}

# ── Master config dict ────────────────────────────────────────────────────────

CFG: dict = {
    # paths
    "CHECKPOINT_DIR"      : "checkpoints",
    "RESULTS_DIR"         : "results",

    # model / inference  (IMG_SIZE must match the saved checkpoint's img_size key)
    "IMG_SIZE"            : 128,
    "DROPOUT"             : 0.3,
    "CONFIDENCE_THRESHOLD": 0.5,
    "TOP_K"               : 3,
    "USE_TTA"             : True,

    # API
    "MAX_BATCH_SIZE"      : 16,
}
