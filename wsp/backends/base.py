from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from wsp.types import EvalResult, Example, MetricFn, PatchManifest, TensorInfo


def sha256_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


class StudentBackend(ABC):
    """Model-agnostic interface for any student neural network."""

    backend_type: str = "base"
    model_id: str = "unknown"

    @abstractmethod
    def list_tensors(self) -> list[TensorInfo]:
        """Return metadata for all editable parameter tensors."""

    @abstractmethod
    def get_tensor(self, name: str, edited: bool = True) -> np.ndarray:
        """Return a copy of the named tensor as a numpy array."""

    @abstractmethod
    def apply_edit(self, name: str, arr: np.ndarray) -> None:
        """Apply a new numpy array to the named tensor in the edited model."""

    @abstractmethod
    def reset_edits(self) -> None:
        """Restore the edited model back to the original base weights."""

    @abstractmethod
    def total_params(self) -> int:
        """Return total parameter count."""

    @abstractmethod
    def forward(
        self,
        examples: list[Example],
        metric: MetricFn,
        split: str = "trainset",
        use_edited: bool = False,
        capture_layers: Optional[list[str]] = None,
        act_dir: Optional[str] = None,
    ) -> EvalResult:
        """Run inference on examples, score with metric, and optionally dump activations."""

    def list_hookable_layers(self) -> list[str]:
        """Return layer names that support activation capture."""
        return []

    def export_patch(
        self,
        out_dir: str,
        holdout_before: Optional[float] = None,
        holdout_after: Optional[float] = None,
        val_before: Optional[float] = None,
        val_after: Optional[float] = None,
    ) -> PatchManifest:
        """Export only changed tensor entries + SHA-256 manifest (dspy-wsp style)."""
        os.makedirs(out_dir, exist_ok=True)
        modified_meta: dict[str, dict[str, Any]] = {}
        npz_payload: dict[str, np.ndarray] = {}
        modified_param_count = 0

        for info in self.list_tensors():
            base_arr = self.get_tensor(info.name, edited=False)
            edit_arr = self.get_tensor(info.name, edited=True)
            diff_mask = base_arr != edit_arr
            num_changed = int(np.count_nonzero(diff_mask))
            if num_changed == 0:
                continue

            modified_param_count += num_changed
            flat_idx = np.flatnonzero(diff_mask).astype(np.int64)
            flat_vals = edit_arr.reshape(-1)[flat_idx]
            safe_key = info.name.replace(".", "__").replace("/", "__")
            npz_payload[f"{safe_key}__idx"] = flat_idx
            npz_payload[f"{safe_key}__vals"] = flat_vals

            modified_meta[info.name] = {
                "shape": list(base_arr.shape),
                "dtype": str(base_arr.dtype),
                "num_changed": num_changed,
                "total_entries": int(base_arr.size),
                "pct_changed": round(100.0 * num_changed / max(1, base_arr.size), 4),
                "base_sha256": sha256_array(base_arr),
                "edited_sha256": sha256_array(edit_arr),
                "key_prefix": safe_key,
            }

        np.savez_compressed(os.path.join(out_dir, "edit.npz"), **npz_payload)

        manifest = PatchManifest(
            base_model_id=self.model_id,
            backend_type=self.backend_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            modified_tensors=modified_meta,
            total_params=self.total_params(),
            modified_params=modified_param_count,
            holdout_score_before=holdout_before,
            holdout_score_after=holdout_after,
            val_score_before=val_before,
            val_score_after=val_after,
        )
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return manifest

    def apply_patch(self, patch_dir: str, verify_hashes: bool = True) -> PatchManifest:
        """Load and apply an exported patch (manifest.json + edit.npz) onto the edited model."""
        with open(os.path.join(patch_dir, "manifest.json"), "r", encoding="utf-8") as f:
            raw = json.load(f)
        npz = np.load(os.path.join(patch_dir, "edit.npz"))

        self.reset_edits()
        for name, meta in raw["modified_tensors"].items():
            base_arr = self.get_tensor(name, edited=False)
            if verify_hashes and sha256_array(base_arr) != meta["base_sha256"]:
                raise ValueError(f"Base tensor SHA-256 mismatch for {name}")
            prefix = meta["key_prefix"]
            idx = npz[f"{prefix}__idx"]
            vals = npz[f"{prefix}__vals"]
            patched = base_arr.copy().reshape(-1)
            patched[idx] = vals
            patched = patched.reshape(base_arr.shape).astype(base_arr.dtype)
            if verify_hashes and sha256_array(patched) != meta["edited_sha256"]:
                raise ValueError(f"Edited tensor SHA-256 mismatch for {name}")
            self.apply_edit(name, patched)

        return PatchManifest(**raw)
