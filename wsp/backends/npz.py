from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np

from wsp.backends.base import StudentBackend
from wsp.types import EvalResult, Example, MetricFn, Prediction, TensorInfo


class NpzBackend(StudentBackend):
    """Lightweight NumPy backend for .npz parameter dictionaries (MLPs / policies)."""

    backend_type = "npz"

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        forward_fn: Callable[[dict[str, np.ndarray], Any], tuple[Any, dict[str, np.ndarray]]],
        model_id: str = "npz-model",
    ) -> None:
        self.model_id = model_id
        self.base_weights = {k: v.copy() for k, v in weights.items()}
        self.edited_weights = {k: v.copy() for k, v in weights.items()}
        self.forward_fn = forward_fn

    def list_tensors(self) -> list[TensorInfo]:
        infos: list[TensorInfo] = []
        for name, arr in self.edited_weights.items():
            infos.append(
                TensorInfo(
                    name=name,
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    size=int(arr.size),
                    min=float(arr.min()) if arr.size else 0.0,
                    max=float(arr.max()) if arr.size else 0.0,
                    mean=float(arr.mean()) if arr.size else 0.0,
                    std=float(arr.std()) if arr.size else 0.0,
                )
            )
        return infos

    def get_tensor(self, name: str, edited: bool = True) -> np.ndarray:
        store = self.edited_weights if edited else self.base_weights
        if name not in store:
            raise KeyError(f"Tensor '{name}' not found.")
        return store[name].copy()

    def apply_edit(self, name: str, arr: np.ndarray) -> None:
        if name not in self.edited_weights:
            raise KeyError(f"Tensor '{name}' not found.")
        orig = self.edited_weights[name]
        if list(arr.shape) != list(orig.shape):
            raise ValueError(f"Shape mismatch for '{name}': expected {list(orig.shape)}, got {list(arr.shape)}")
        self.edited_weights[name] = np.asarray(arr, dtype=orig.dtype).copy()

    def reset_edits(self) -> None:
        self.edited_weights = {k: v.copy() for k, v in self.base_weights.items()}

    def total_params(self) -> int:
        return sum(int(v.size) for v in self.base_weights.values())

    def forward(
        self,
        examples: list[Example],
        metric: MetricFn,
        split: str = "trainset",
        use_edited: bool = False,
        capture_layers: Optional[list[str]] = None,
        act_dir: Optional[str] = None,
    ) -> EvalResult:
        store = self.edited_weights if use_edited else self.base_weights
        captured: dict[str, list[np.ndarray]] = {l: [] for l in (capture_layers or [])}
        predictions: list[Prediction] = []
        total_score = 0.0

        for ex in examples:
            out, acts = self.forward_fn(store, ex.input)
            for l in captured:
                if l in acts:
                    captured[l].append(acts[l])
            score, feedback = metric(out, ex.target, ex)
            total_score += float(score)
            predictions.append(
                Prediction(
                    example_id=ex.id,
                    output=out.tolist() if isinstance(out, np.ndarray) else out,
                    score=float(score),
                    feedback=feedback,
                    correct=bool(score >= 1.0),
                )
            )

        activation_files: dict[str, str] = {}
        if capture_layers and act_dir:
            os.makedirs(act_dir, exist_ok=True)
            for lname, chunks in captured.items():
                if not chunks:
                    continue
                arr = np.stack(chunks, axis=0)
                safe = lname.replace(".", "_")
                path = os.path.join(act_dir, f"{safe}.npy")
                np.save(path, arr)
                activation_files[lname] = path

        return EvalResult(
            model_tag="edited" if use_edited else "original",
            split=split,
            score=total_score / max(1, len(examples)),
            num_examples=len(examples),
            predictions=predictions,
            activation_files=activation_files,
        )
