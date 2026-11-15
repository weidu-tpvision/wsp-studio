from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wsp.optimizer import WSPOptimizer
from wsp.tasks.builtin import list_builtin_tasks, load_builtin_task
from wsp.tasks.custom import load_task_from_python_file
from wsp.writers import BedrockWriter, GeminiWriter, MockWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RUNS_ROOT = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_ROOT, exist_ok=True)

app = FastAPI(title="WSP Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory registry of active/completed runs
RUNS: dict[str, dict[str, Any]] = {}


class StartRunRequest(BaseModel):
    task_id: str = "cnn_demo"
    task_file: Optional[str] = None
    writer_provider: str = "gemini"  # gemini | bedrock | mock
    writer_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    aws_region: Optional[str] = None
    hf_token: Optional[str] = None
    max_passes: int = 15


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/tasks")
def get_tasks():
    return {"tasks": list_builtin_tasks()}


@app.get("/api/runs")
def list_runs():
    return {
        "runs": [
            {
                "run_id": rid,
                "status": r["status"],
                "task_id": r["task_id"],
                "writer": r["writer"],
                "turns_count": len(r["turns"]),
                "val_before": r.get("val_before"),
                "val_after": r.get("val_after"),
            }
            for rid, r in RUNS.items()
        ]
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNS[run_id]


def _execute_run(run_id: str, req: StartRunRequest) -> None:
    state = RUNS[run_id]
    try:
        if req.hf_token:
            os.environ["HF_TOKEN"] = req.hf_token

        if req.task_file:
            backend, dataset, metric, desc = load_task_from_python_file(req.task_file)
        else:
            backend, dataset, metric, desc = load_builtin_task(req.task_id)

        state["model_id"] = backend.model_id
        state["total_params"] = backend.total_params()
        state["tensors"] = [t.to_dict() for t in backend.list_tensors()[:40]]

        if req.writer_provider == "gemini":
            writer = GeminiWriter(
                model=req.writer_model or "gemini-2.5-pro",
                api_key=req.gemini_api_key or None,
            )
        elif req.writer_provider == "bedrock":
            writer = BedrockWriter(
                model_id=req.writer_model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                region_name=req.aws_region or None,
            )
        else:
            writer = MockWriter()

        def on_turn(rec):
            state["turns"].append(rec.to_dict())
            state["passes_left"] = rec.passes_left

        run_dir = os.path.join(RUNS_ROOT, run_id)
        opt = WSPOptimizer(
            backend=backend,
            writer=writer,
            metric=metric,
            task_description=desc,
            max_passes=req.max_passes,
            run_dir=run_dir,
            on_turn=on_turn,
        )
        res = opt.compile(dataset)
        state["status"] = "completed"
        state["holdout_before"] = res.holdout_score_before
        state["holdout_after"] = res.holdout_score_after
        state["val_before"] = res.val_score_before
        state["val_after"] = res.val_score_after
        state["manifest"] = res.manifest.to_dict()
        state["run_dir"] = res.run_dir
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)


@app.post("/api/runs")
def start_run(req: StartRunRequest):
    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {
        "run_id": run_id,
        "status": "running",
        "task_id": req.task_file or req.task_id,
        "writer": f"{req.writer_provider}:{req.writer_model or 'default'}",
        "passes_left": req.max_passes,
        "max_passes": req.max_passes,
        "turns": [],
        "error": None,
    }
    t = threading.Thread(target=_execute_run, args=(run_id, req), daemon=True)
    t.start()
    return {"run_id": run_id}
