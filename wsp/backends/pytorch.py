from __future__ import annotations

import copy
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from wsp.backends.base import StudentBackend
from wsp.types import EvalResult, Example, MetricFn, Prediction, TensorInfo


class PyTorchBackend(StudentBackend):
    """Generic backend for any PyTorch nn.Module (CNNs, ResNets, MLPs, etc.)."""

    backend_type = "pytorch"

    def __init__(
        self,
        model: nn.Module,
        model_id: str = "pytorch-model",
        predict_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.base_model = copy.deepcopy(model).to(self.device).eval()
        self.edited_model = copy.deepcopy(model).to(self.device).eval()
        self.predict_fn = predict_fn or self._default_predict

    @staticmethod
    def _default_predict(model: nn.Module, x: Any) -> Any:
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            tensor = x.float()
        else:
            tensor = torch.tensor(x, dtype=torch.float32)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            # Single CHW image -> NCHW
            tensor = tensor.unsqueeze(0)

        out = model(tensor)
        if isinstance(out, torch.Tensor):
            if out.ndim >= 2 and out.shape[0] == 1:
                out = out.squeeze(0)
            return out.detach().cpu().numpy()
        return out

    def _model(self, edited: bool) -> nn.Module:
        return self.edited_model if edited else self.base_model

    def list_tensors(self) -> list[TensorInfo]:
        infos: list[TensorInfo] = []
        for name, param in self.edited_model.named_parameters():
            arr = param.detach().cpu().numpy()
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

    def list_hookable_layers(self) -> list[str]:
        return [name for name, _ in self.edited_model.named_modules() if name]

    def get_tensor(self, name: str, edited: bool = True) -> np.ndarray:
        params = dict(self._model(edited).named_parameters())
        if name not in params:
            raise KeyError(f"Tensor '{name}' not found. Available: {list(params.keys())[:10]}")
        return params[name].detach().cpu().numpy().copy()

    def apply_edit(self, name: str, arr: np.ndarray) -> None:
        params = dict(self.edited_model.named_parameters())
        if name not in params:
            raise KeyError(f"Tensor '{name}' not found.")
        param = params[name]
        if list(arr.shape) != list(param.shape):
            raise ValueError(f"Shape mismatch for '{name}': expected {list(param.shape)}, got {list(arr.shape)}")
        with torch.no_grad():
            new_t = torch.from_numpy(np.asarray(arr)).to(device=param.device, dtype=param.dtype)
            param.copy_(new_t)

    def reset_edits(self) -> None:
        self.edited_model.load_state_dict(copy.deepcopy(self.base_model.state_dict()))

    def total_params(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())

    def forward(
        self,
        examples: list[Example],
        metric: MetricFn,
        split: str = "trainset",
        use_edited: bool = False,
        capture_layers: Optional[list[str]] = None,
        act_dir: Optional[str] = None,
    ) -> EvalResult:
        model = self._model(use_edited)
        model.eval()

        captured: dict[str, list[np.ndarray]] = {}
        handles: list[Any] = []
        if capture_layers:
            modules = dict(model.named_modules())
            for layer_name in capture_layers:
                if layer_name not in modules:
                    raise KeyError(f"Layer '{layer_name}' not found for activation capture.")
                captured[layer_name] = []

                def _make_hook(lname: str):
                    def _hook(_mod, _inp, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        if isinstance(out, torch.Tensor):
                            captured[lname].append(out.detach().cpu().numpy())
                    return _hook

                handles.append(modules[layer_name].register_forward_hook(_make_hook(layer_name)))

        predictions: list[Prediction] = []
        total_score = 0.0
        try:
            with torch.no_grad():
                for ex in examples:
                    out = self.predict_fn(model, ex.input)
                    score, feedback = metric(out, ex.target, ex)
                    total_score += float(score)
                    pred_label = int(np.argmax(out)) if isinstance(out, np.ndarray) and out.ndim == 1 else out
                    predictions.append(
                        Prediction(
                            example_id=ex.id,
                            output=pred_label if not isinstance(pred_label, np.ndarray) else pred_label.tolist(),
                            score=float(score),
                            feedback=feedback,
                            correct=bool(score >= 1.0),
                        )
                    )
        finally:
            for h in handles:
                h.remove()

        activation_files: dict[str, str] = {}
        if capture_layers and act_dir:
            os.makedirs(act_dir, exist_ok=True)
            for lname, chunks in captured.items():
                if not chunks:
                    continue
                arr = np.concatenate(chunks, axis=0) if chunks[0].ndim >= 1 else np.array(chunks)
                safe = lname.replace(".", "_")
                path = os.path.join(act_dir, f"{safe}.npy")
                np.save(path, arr)
                activation_files[lname] = path

        avg_score = total_score / max(1, len(examples))
        return EvalResult(
            model_tag="edited" if use_edited else "original",
            split=split,
            score=avg_score,
            num_examples=len(examples),
            predictions=predictions,
            activation_files=activation_files,
        )
