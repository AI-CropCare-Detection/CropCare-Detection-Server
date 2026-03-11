"""
predictor.py  –  Inference engine for plant-disease classification.
Wraps PlantDiseasePredictor so the rest of the server never touches
PyTorch / Albumentations directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, DEVICE, NORM
from model_arch import YOLOv7Classifier        # your existing architecture file
from tta import tta_transforms                  # your existing TTA transforms


# ── Predictor ────────────────────────────────────────────────────────────────

class PlantDiseasePredictor:
    """
    Production inference engine with TTA support.

    Accepted image sources
    ----------------------
    - File path  : str | pathlib.Path
    - PIL image  : PIL.Image.Image
    - NumPy array: np.ndarray  (H×W×3, uint8, RGB)

    Returns
    -------
    dict with keys:
        class, class_id, confidence, top_k, all_probs
    """

    def __init__(self, ckpt_path: str | Path, device: str | None = None) -> None:
        self.device = device or DEVICE
        ckpt_path = Path(ckpt_path)

        # Try loading as TorchScript first, then fall back to regular checkpoint
        try:
            self.model = torch.jit.load(ckpt_path, map_location=self.device)
            # Load metadata from JSON file
            metadata_path = ckpt_path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.class_names = metadata["class_names"]
                self.nc = metadata["num_classes"]
                self.isz = metadata["img_size"]
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        except (RuntimeError, NotImplementedError):
            # Fall back to regular torch.load for standard checkpoints
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.class_names: list[str] = ckpt["class_names"]
            self.nc:   int = ckpt["num_classes"]
            self.isz:  int = ckpt["img_size"]
            
            self.model = YOLOv7Classifier(self.nc, dropout=0.0)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device).eval()
            return

        self.model.to(self.device).eval()

        self.base_tfm = A.Compose([
            A.Resize(self.isz, self.isz),
            A.Normalize(**NORM),
            ToTensorV2(),
        ])

        print(f"[Predictor] Ready — {self.nc} classes on {self.device}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load(self, src: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Convert any supported source to an RGB uint8 NumPy array."""
        if isinstance(src, (str, Path)):
            return np.array(Image.open(src).convert("RGB"))
        if isinstance(src, Image.Image):
            return np.array(src.convert("RGB"))
        return src  # already np.ndarray

    @torch.no_grad()
    def _run(self, img: np.ndarray, use_tta: bool) -> np.ndarray:
        """Return a probability vector (np.ndarray, shape [nc])."""
        if use_tta:
            probs = torch.stack([
                F.softmax(
                    self.model(
                        t(image=img)["image"].unsqueeze(0).to(self.device)
                    ), dim=1
                )
                for t in tta_transforms
            ]).mean(0)
        else:
            tensor = self.base_tfm(image=img)["image"].unsqueeze(0).to(self.device)
            probs  = F.softmax(self.model(tensor), dim=1)

        return probs.squeeze().cpu().numpy()

    # ── public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        src: Union[str, Path, Image.Image, np.ndarray],
        use_tta: bool = True,
        top_k:   int  = 3,
    ) -> dict:
        """Return prediction dict for a single image."""
        img = self._load(src)
        p   = self._run(img, use_tta)
        pi  = int(p.argmax())
        tk  = min(top_k, self.nc)

        return {
            "class"      : self.class_names[pi],
            "class_id"   : pi,
            "confidence" : float(p[pi]),
            "top_k"      : [
                {"class": self.class_names[i], "confidence": float(p[i])}
                for i in p.argsort()[::-1][:tk]
            ],
            "all_probs"  : {
                self.class_names[i]: float(p[i]) for i in range(self.nc)
            },
        }

    def predict_batch(
        self,
        sources: list,
        use_tta: bool = False,
    ) -> list[dict]:
        """Run predict() over a list of sources."""
        return [
            self.predict(s, use_tta=use_tta)
            for s in tqdm(sources, desc="Batch inference")
        ]

    def visualise(
        self,
        src: Union[str, Path, Image.Image, np.ndarray],
        use_tta: bool = True,
        save: bool = True,
    ) -> dict:
        """Predict and plot a side-by-side result chart."""
        img = self._load(src)
        res = self.predict(src, use_tta=use_tta)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.imshow(cv2.resize(img, (self.isz, self.isz)))
        colour = (
            "green" if res["confidence"] >= CFG["CONFIDENCE_THRESHOLD"] else "orange"
        )
        ax1.set_title(
            f"Prediction: {res['class']}\nConfidence: {res['confidence']:.1%}",
            fontsize=13, fontweight="bold", color=colour,
        )
        ax1.axis("off")

        classes  = list(res["all_probs"].keys())
        values   = list(res["all_probs"].values())
        bar_cols = [
            "green" if c == res["class"] else "steelblue" for c in classes
        ]
        bars = ax2.barh(classes, values, color=bar_cols)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        ax2.axvline(
            CFG["CONFIDENCE_THRESHOLD"],
            color="red", ls="--", lw=1.2, label="Threshold",
        )
        ax2.set_title("Class Probabilities")
        ax2.legend()

        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9,
            )

        plt.tight_layout()

        if save:
            out_path = Path(CFG["RESULTS_DIR"]) / "inference_result.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")

        plt.show()
        return res


# ── Singleton helper (used by the FastAPI app) ────────────────────────────────

_predictor: PlantDiseasePredictor | None = None


def get_predictor() -> PlantDiseasePredictor:
    """
    Return the module-level singleton, initialising it on first call.
    Allows FastAPI to use dependency injection without re-loading the model.
    """
    global _predictor
    if _predictor is None:
        ckpt = Path(CFG["CHECKPOINT_DIR"]) / "yolov7_plant_disease.torchscript.pt"
        _predictor = PlantDiseasePredictor(ckpt)
    return _predictor
