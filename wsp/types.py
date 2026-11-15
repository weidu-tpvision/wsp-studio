from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


@dataclass
class Example:
    """A single task example (input + expected target + optional metadata)."""
    id: str
    input: Any
    target: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Output of running the student model on one Example."""
    example_id: str
    output: Any
    score: float = 0.0
    feedback: str = ""
    correct: bool = False


@dataclass
class EvalResult:
    """Aggregated evaluation result over a split of examples."""
    model_tag: str  # "original" | "edited"
    split: str      # "trainset" | "holdout" | "valset"
    score: float
    num_examples: int
    predictions: list[Prediction] = field(default_factory=list)
    activation_files: dict[str, str] = field(default_factory=dict)  # layer -> .npy path

    def summary_dict(self, include_examples: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model_tag,
            "split": self.split,
            "score": round(self.score, 4),
            "num_examples": self.num_examples,
            "activation_files": self.activation_files,
        }
        if include_examples:
            out["examples"] = [
                {
                    "id": p.example_id,
                    "output": p.output,
                    "score": p.score,
                    "correct": p.correct,
                    "feedback": p.feedback,
                }
                for p in self.predictions
            ]
        return out


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    size: int
    min: float
    max: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    turn: int
    passes_left: int
    reason: str
    action_type: str  # "forward" | "python" | "finish" | "error"
    action_payload: str
    observation: str
    holdout_score: Optional[float] = None
    is_best: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchManifest:
    base_model_id: str
    backend_type: str
    created_at: str
    modified_tensors: dict[str, dict[str, Any]]
    total_params: int
    modified_params: int
    holdout_score_before: Optional[float] = None
    holdout_score_after: Optional[float] = None
    val_score_before: Optional[float] = None
    val_score_after: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    task_name: str
    max_passes: int = 25
    holdout_ratio: float = 0.25
    seed: int = 42
    run_dir: str = "runs/latest"


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    val_score_before: float
    val_score_after: float
    holdout_score_before: float
    holdout_score_after: float
    passes_used: int
    manifest: PatchManifest
    turns: list[TurnRecord] = field(default_factory=list)
    final_report: str = ""


MetricFn = Callable[[Any, Any, Example], tuple[float, str]]
