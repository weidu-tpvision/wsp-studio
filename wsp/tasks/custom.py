from __future__ import annotations

import importlib.util
import os
from typing import Any

import torch
import torch.nn as nn

from wsp.backends.pytorch import PyTorchBackend
from wsp.types import Example, MetricFn


def load_pytorch_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    model_id: str = "custom-pytorch",
    predict_fn: Any = None,
) -> PyTorchBackend:
    """Load a user-provided PyTorch .pt/.pth state_dict into an nn.Module."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return PyTorchBackend(model, model_id=model_id, predict_fn=predict_fn)


def load_task_from_python_file(py_path: str) -> tuple[Any, list[Example], MetricFn, str]:
    """
    Load a user-provided task module defining `load_task() -> (backend, dataset, metric, description)`.
    """
    abs_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location("user_wsp_task", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_task"):
        raise AttributeError(f"{abs_path} must define a `load_task()` function.")
    return mod.load_task()
