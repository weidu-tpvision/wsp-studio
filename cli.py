from __future__ import annotations

import argparse
import os
import sys

from wsp.optimizer import WSPOptimizer
from wsp.tasks.builtin import load_builtin_task
from wsp.tasks.custom import load_task_from_python_file
from wsp.writers import BedrockWriter, GeminiWriter, MockWriter


def build_writer(args: argparse.Namespace):
    if args.writer == "gemini":
        return GeminiWriter(model=args.model or "gemini-2.5-pro")
    if args.writer == "bedrock":
        return BedrockWriter(
            model_id=args.model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name=args.region,
        )
    if args.writer == "mock":
        return MockWriter()
    raise ValueError(f"Unknown writer: {args.writer}")


def main() -> int:
    parser = argparse.ArgumentParser(description="WSP Studio CLI — General-Purpose Weight-Space Programming")
    parser.add_argument("--task", default="cnn_demo", help="Builtin task id: cnn_demo | functiongemma_tools | qwen_math")
    parser.add_argument("--task-file", default=None, help="Path to a custom Python file defining load_task()")
    parser.add_argument("--writer", default="gemini", choices=["gemini", "bedrock", "mock"], help="Writer LLM provider")
    parser.add_argument("--model", default=None, help="Writer model ID (e.g. gemini-2.5-pro or Bedrock modelId)")
    parser.add_argument("--region", default=None, help="AWS region for Bedrock")
    parser.add_argument("--max-passes", type=int, default=25, help="Forward-pass budget")
    parser.add_argument("--run-dir", default=None, help="Output directory for run artifacts and patch")
    args = parser.parse_args()

    if args.task_file:
        backend, dataset, metric, desc = load_task_from_python_file(args.task_file)
    else:
        backend, dataset, metric, desc = load_builtin_task(args.task)

    writer = build_writer(args)

    def on_turn(rec):
        badge = " ★ NEW BEST" if rec.is_best else ""
        ho = f" | holdout={rec.holdout_score:.4f}" if rec.holdout_score is not None else ""
        print(f"\n[Turn {rec.turn} | passes_left={rec.passes_left} | {rec.action_type}{ho}{badge}]")
        print(f"REASON: {rec.reason}")

    opt = WSPOptimizer(
        backend=backend,
        writer=writer,
        metric=metric,
        task_description=desc,
        max_passes=args.max_passes,
        run_dir=args.run_dir,
        on_turn=on_turn,
    )
    res = opt.compile(dataset)
    print("\n" + "=" * 60)
    print(f"Run complete: {res.run_dir}")
    print(f"Holdout: {res.holdout_score_before:.4f} -> {res.holdout_score_after:.4f}")
    print(f"Valset : {res.val_score_before:.4f} -> {res.val_score_after:.4f}")
    print(f"Modified params: {res.manifest.modified_params:,} / {res.manifest.total_params:,}")
    print(f"Patch saved to: {os.path.join(res.run_dir, 'patch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
