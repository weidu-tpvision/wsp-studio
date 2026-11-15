from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from wsp.backends.base import StudentBackend


_PRELUDE_TEMPLATE = '''
import json
import os
import numpy as np

_SANDBOX_ROOT = {sandbox_root!r}
_BASE_DIR = os.path.join(_SANDBOX_ROOT, "weights_base")
_EDITED_DIR = os.path.join(_SANDBOX_ROOT, "edited")
_INDEX_PATH = os.path.join(_SANDBOX_ROOT, "weight_index.json")

with open(_INDEX_PATH, "r", encoding="utf-8") as _f:
    _WEIGHT_INDEX = json.load(_f)

def list_weights():
    """Return dict of tensor_name -> metadata (shape, dtype, size, min, max, mean, std)."""
    return _WEIGHT_INDEX

def _safe_file(name: str) -> str:
    if name not in _WEIGHT_INDEX:
        raise KeyError(f"Unknown weight tensor: {{name!r}}. Call list_weights() to see valid names.")
    return _WEIGHT_INDEX[name]["file"]

def load_weight(name: str, edited: bool = True) -> np.ndarray:
    """Load a weight tensor as a numpy array (reads edited version if present, else base)."""
    fname = _safe_file(name)
    edited_path = os.path.join(_EDITED_DIR, fname)
    base_path = os.path.join(_BASE_DIR, fname)
    path = edited_path if (edited and os.path.exists(edited_path)) else base_path
    return np.load(path).copy()

def save_edit(name: str, arr: np.ndarray) -> str:
    """Save an edited weight tensor to the sandbox edited/ directory."""
    fname = _safe_file(name)
    expected_shape = tuple(_WEIGHT_INDEX[name]["shape"])
    arr = np.asarray(arr)
    if arr.shape != expected_shape:
        raise ValueError(f"Shape mismatch for {{name}}: expected {{expected_shape}}, got {{arr.shape}}")
    os.makedirs(_EDITED_DIR, exist_ok=True)
    out_path = os.path.join(_EDITED_DIR, fname)
    np.save(out_path, arr)
    return f"Saved edit for {{name}} ({{arr.size}} entries) -> edited/{{fname}}"

def reset_edits() -> str:
    """Remove all pending edits and revert to base weights."""
    for f in os.listdir(_EDITED_DIR):
        if f.endswith(".npy"):
            os.remove(os.path.join(_EDITED_DIR, f))
    return "All edits reset to base weights."

def load_act(path: str) -> np.ndarray:
    """Load a captured activation .npy file from the sandbox."""
    full = path if os.path.isabs(path) else os.path.join(_SANDBOX_ROOT, path)
    return np.load(full)

def save_note(text: str) -> None:
    with open(os.path.join(_SANDBOX_ROOT, "notes.md"), "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\\n")
'''


class SandboxedKernel:
    """Isolated Python execution workspace for Weight-Space Programming."""

    def __init__(self, backend: StudentBackend, run_dir: str, timeout_sec: int = 60) -> None:
        self.backend = backend
        self.run_dir = os.path.abspath(run_dir)
        self.work_dir = os.path.join(self.run_dir, "work")
        self.base_dir = os.path.join(self.run_dir, "weights_base")
        self.edited_dir = os.path.join(self.run_dir, "edited")
        self.best_dir = os.path.join(self.run_dir, "best")
        self.timeout_sec = timeout_sec
        self.weight_index: dict[str, dict[str, Any]] = {}
        self._init_workspace()

    def _init_workspace(self) -> None:
        for d in (self.run_dir, self.work_dir, self.base_dir, self.edited_dir, self.best_dir):
            os.makedirs(d, exist_ok=True)

        self.weight_index = {}
        for idx, info in enumerate(self.backend.list_tensors()):
            fname = f"w_{idx:04d}.npy"
            arr = self.backend.get_tensor(info.name, edited=False)
            np.save(os.path.join(self.base_dir, fname), arr)
            self.weight_index[info.name] = {
                "file": fname,
                "shape": info.shape,
                "dtype": info.dtype,
                "size": info.size,
                "min": round(info.min, 6),
                "max": round(info.max, 6),
                "mean": round(info.mean, 6),
                "std": round(info.std, 6),
            }

        with open(os.path.join(self.run_dir, "weight_index.json"), "w", encoding="utf-8") as f:
            json.dump(self.weight_index, f, indent=2)

    def snapshot_best(self) -> None:
        """Copy current edited/ weights into best/."""
        for f in os.listdir(self.best_dir):
            os.remove(os.path.join(self.best_dir, f))
        for f in os.listdir(self.edited_dir):
            if f.endswith(".npy"):
                shutil.copy2(os.path.join(self.edited_dir, f), os.path.join(self.best_dir, f))

    def restore_best_to_backend(self) -> None:
        """Load best/ snapshot back into the StudentBackend."""
        self.backend.reset_edits()
        for name, meta in self.weight_index.items():
            best_path = os.path.join(self.best_dir, meta["file"])
            if os.path.exists(best_path):
                self.backend.apply_edit(name, np.load(best_path))

    def sync_edited_to_backend(self) -> list[str]:
        """Push all .npy files in edited/ into StudentBackend.edited_model."""
        self.backend.reset_edits()
        applied: list[str] = []
        for name, meta in self.weight_index.items():
            edit_path = os.path.join(self.edited_dir, meta["file"])
            if os.path.exists(edit_path):
                self.backend.apply_edit(name, np.load(edit_path))
                applied.append(name)
        return applied

    def run_python(self, code: str) -> str:
        """Execute Writer Python code in an isolated subprocess and sync edits."""
        # Guardrail: forbid gradient-descent / optimizer training loops
        forbidden = [".backward(", "torch.optim", "SFTTrainer", "Adam(", "SGD("]
        for tok in forbidden:
            if tok in code:
                return f"ERROR: Gradient-based training ({tok}) is forbidden in Weight-Space Programming."

        script_path = os.path.join(self.work_dir, "_turn_script.py")
        prelude = _PRELUDE_TEMPLATE.format(sandbox_root=self.run_dir)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(prelude + "\n\n" + code + "\n")

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.work_dir,
            "PYTHONNOUSERSITE": "1",
            # Block outbound network inside sandboxed Python
            "HTTP_PROXY": "http://127.0.0.1:0",
            "HTTPS_PROXY": "http://127.0.0.1:0",
        }
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=self.work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: Python execution timed out after {self.timeout_sec}s."

        applied = self.sync_edited_to_backend()
        out_parts = []
        if proc.stdout.strip():
            out_parts.append(proc.stdout.strip())
        if proc.stderr.strip():
            out_parts.append(f"STDERR:\n{proc.stderr.strip()}")
        if applied:
            out_parts.append(f"[Synced {len(applied)} edited tensor(s) to student: {applied}]")
        if proc.returncode != 0:
            out_parts.append(f"[Exit code: {proc.returncode}]")
        return "\n".join(out_parts) if out_parts else "(no output)"
