from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from wsp.backends.base import StudentBackend
from wsp.sandbox import SandboxedKernel
from wsp.types import (
    Example,
    MetricFn,
    RunResult,
    TurnRecord,
)
from wsp.writers.base import WriterLLM, parse_writer_response


def split_dataset(
    dataset: list[Example],
    holdout_ratio: float = 0.25,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split dataset into (trainset, holdout, valset)."""
    items = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    n_hold = max(1, int(round(n * holdout_ratio))) if n - n_val >= 2 else 0
    valset = items[:n_val]
    holdout = items[n_val : n_val + n_hold]
    trainset = items[n_val + n_hold :]
    return trainset, holdout, valset


class WSPOptimizer:
    """General-purpose Weight-Space Programming optimizer."""

    def __init__(
        self,
        backend: StudentBackend,
        writer: WriterLLM,
        metric: MetricFn,
        task_description: str,
        max_passes: int = 25,
        holdout_ratio: float = 0.25,
        val_ratio: float = 0.25,
        seed: int = 42,
        run_dir: Optional[str] = None,
        on_turn: Optional[Callable[[TurnRecord], None]] = None,
    ) -> None:
        self.backend = backend
        self.writer = writer
        self.metric = metric
        self.task_description = task_description
        self.max_passes = max_passes
        self.holdout_ratio = holdout_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.run_dir = os.path.abspath(run_dir or os.path.join("runs", self.run_id))
        self.on_turn = on_turn

    def _build_system_prompt(self, trainset: list[Example], holdout: list[Example]) -> str:
        tensors = self.backend.list_tensors()
        tensor_lines = [
            f"  - {t.name}: shape={t.shape}, dtype={t.dtype}, min={t.min:.4f}, max={t.max:.4f}, std={t.std:.4f}"
            for t in tensors[:80]
        ]
        if len(tensors) > 80:
            tensor_lines.append(f"  ... ({len(tensors) - 80} more tensors; call list_weights() in python)")

        hookable = self.backend.list_hookable_layers()
        hook_str = ", ".join(hookable[:40]) + (" ..." if len(hookable) > 40 else "")

        train_ids = [ex.id for ex in trainset]

        return f"""You are a Weight-Space Programmer: an expert neural network engineer optimizing a student model's weights directly without gradient descent.

TASK:
{self.task_description}

STUDENT MODEL:
- Model ID: {self.backend.model_id}
- Backend: {self.backend.backend_type}
- Total parameters: {self.backend.total_params():,}
- Editable tensors:
{chr(10).join(tensor_lines)}

- Hookable layers for activation capture:
  {hook_str or "(none)"}

DATA SPLITS:
- `trainset` ({len(trainset)} examples, IDs: {train_ids}): You may inspect outputs, errors, and capture activations.
- `holdout` ({len(holdout)} examples): Blind check set. You may score `edited` vs `original` on `holdout`, but you will only receive the aggregate score (no raw examples).

PROTOCOL (STRICT):
You have a budget of {self.max_passes} `forward` passes.
In every turn, output a short `REASON:` explaining your hypothesis, followed by **EXACTLY ONE** fenced code block of type `forward`, `python`, or `finish`:

1. `forward` block (costs 1 pass):
```forward
{{"model": "original" | "edited" | "both", "split": "trainset" | "holdout", "examples": ["id1", ...], "capture_layers": ["layer_name"]}}
```
- `examples` is optional (omit to run the full split).
- `capture_layers` is optional; only available on `trainset` and saves `.npy` activation files you can inspect in `python`.

2. `python` block (free, does NOT cost a forward pass):
```python
# Available helpers: list_weights(), load_weight(name, edited=True), save_edit(name, arr), reset_edits(), load_act(path), save_note(text)
```
- No gradient descent / `.backward()` / optimizers allowed.

3. `finish` block (ends optimization early):
```finish
{{"summary": "Brief summary of findings and edits."}}
```
"""

    def compile(
        self,
        dataset: list[Example],
        trainset: Optional[list[Example]] = None,
        holdout: Optional[list[Example]] = None,
        valset: Optional[list[Example]] = None,
    ) -> RunResult:
        if trainset is None or holdout is None or valset is None:
            tr, ho, va = split_dataset(dataset, self.holdout_ratio, self.val_ratio, self.seed)
            trainset = trainset if trainset is not None else tr
            holdout = holdout if holdout is not None else ho
            valset = valset if valset is not None else va

        train_by_id = {ex.id: ex for ex in trainset}
        kernel = SandboxedKernel(self.backend, self.run_dir)
        system_prompt = self._build_system_prompt(trainset, holdout)

        # Baseline holdout & valset scores (before optimization)
        check_split = holdout if holdout else trainset
        base_holdout_res = self.backend.forward(check_split, self.metric, split="holdout", use_edited=False)
        holdout_before = base_holdout_res.score
        best_holdout = holdout_before
        kernel.snapshot_best()

        passes_left = self.max_passes
        turns: list[TurnRecord] = []
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Optimization started. Budget: {passes_left} forward passes.\n"
                    f"Baseline holdout score: {holdout_before:.4f}.\n"
                    f"Begin by running a baseline `forward` on `trainset` or inspecting weights in `python`."
                ),
            }
        ]
        final_report = ""
        turn_idx = 0

        while passes_left > 0:
            turn_idx += 1
            raw_reply = self.writer.generate(system_prompt, messages)
            messages.append({"role": "assistant", "content": raw_reply})

            try:
                parsed = parse_writer_response(raw_reply)
            except ValueError as e:
                obs = f"PARSE ERROR: {e}"
                rec = TurnRecord(
                    turn=turn_idx,
                    passes_left=passes_left,
                    reason="(parse error)",
                    action_type="error",
                    action_payload=raw_reply,
                    observation=obs,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                turns.append(rec)
                if self.on_turn:
                    self.on_turn(rec)
                messages.append({"role": "user", "content": obs})
                continue

            if parsed.action_type == "finish":
                final_report = parsed.payload
                rec = TurnRecord(
                    turn=turn_idx,
                    passes_left=passes_left,
                    reason=parsed.reason,
                    action_type="finish",
                    action_payload=parsed.payload,
                    observation="Optimization finished by writer.",
                    holdout_score=best_holdout,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                turns.append(rec)
                if self.on_turn:
                    self.on_turn(rec)
                break

            holdout_score_now: Optional[float] = None
            is_best = False

            if parsed.action_type == "python":
                obs = kernel.run_python(parsed.payload)
            elif parsed.action_type == "forward":
                passes_left -= 1
                try:
                    cmd = json.loads(parsed.payload)
                except json.JSONDecodeError as e:
                    obs = f"JSON ERROR in forward block: {e}"
                    cmd = None

                if cmd is not None:
                    model_choice = cmd.get("model", "edited")
                    split_name = cmd.get("split", "trainset")
                    capture_layers = cmd.get("capture_layers")
                    ex_ids = cmd.get("examples")

                    if split_name == "holdout":
                        target_examples = holdout if holdout else trainset
                        capture_layers = None  # Never leak raw holdout activations
                        include_examples = False
                    else:
                        if ex_ids:
                            target_examples = [train_by_id[eid] for eid in ex_ids if eid in train_by_id]
                            if not target_examples:
                                target_examples = trainset
                        else:
                            target_examples = trainset
                        include_examples = True

                    act_dir = os.path.join(self.run_dir, f"fwd_{turn_idx}") if capture_layers else None
                    results_payload: dict[str, Any] = {"passes_left": passes_left}

                    if model_choice in ("original", "both"):
                        res_orig = self.backend.forward(
                            target_examples,
                            self.metric,
                            split=split_name,
                            use_edited=False,
                            capture_layers=capture_layers if model_choice == "original" else None,
                            act_dir=act_dir,
                        )
                        results_payload["original"] = res_orig.summary_dict(include_examples=include_examples)

                    if model_choice in ("edited", "both"):
                        res_edit = self.backend.forward(
                            target_examples,
                            self.metric,
                            split=split_name,
                            use_edited=True,
                            capture_layers=capture_layers,
                            act_dir=act_dir,
                        )
                        results_payload["edited"] = res_edit.summary_dict(include_examples=include_examples)

                        # Score on blind holdout to track best checkpoint
                        ho_eval = (
                            res_edit
                            if split_name == "holdout"
                            else self.backend.forward(check_split, self.metric, split="holdout", use_edited=True)
                        )
                        holdout_score_now = ho_eval.score
                        results_payload["blind_holdout_score"] = round(holdout_score_now, 4)
                        if holdout_score_now > best_holdout:
                            best_holdout = holdout_score_now
                            kernel.snapshot_best()
                            is_best = True
                            results_payload["new_best"] = True

                    obs = json.dumps(results_payload, indent=2)
            else:
                obs = f"Unknown action type: {parsed.action_type}"

            rec = TurnRecord(
                turn=turn_idx,
                passes_left=passes_left,
                reason=parsed.reason,
                action_type=parsed.action_type,
                action_payload=parsed.payload,
                observation=obs,
                holdout_score=holdout_score_now,
                is_best=is_best,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            turns.append(rec)
            if self.on_turn:
                self.on_turn(rec)
            messages.append({"role": "user", "content": f"OBSERVATION (passes_left={passes_left}):\n{obs}"})

        # Restore best edited weights and run final blind valset evaluation
        kernel.restore_best_to_backend()
        eval_val = valset if valset else check_split
        val_before_res = self.backend.forward(eval_val, self.metric, split="valset", use_edited=False)
        val_after_res = self.backend.forward(eval_val, self.metric, split="valset", use_edited=True)

        patch_dir = os.path.join(self.run_dir, "patch")
        manifest = self.backend.export_patch(
            patch_dir,
            holdout_before=holdout_before,
            holdout_after=best_holdout,
            val_before=val_before_res.score,
            val_after=val_after_res.score,
        )

        with open(os.path.join(self.run_dir, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in turns], f, indent=2)

        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            val_score_before=val_before_res.score,
            val_score_after=val_after_res.score,
            holdout_score_before=holdout_before,
            holdout_score_after=best_holdout,
            passes_used=self.max_passes - passes_left,
            manifest=manifest,
            turns=turns,
            final_report=final_report,
        )
