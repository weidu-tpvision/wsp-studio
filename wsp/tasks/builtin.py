from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from wsp.backends.base import StudentBackend
from wsp.backends.hf import HFBackend
from wsp.backends.pytorch import PyTorchBackend
from wsp.types import Example, MetricFn


class SmallPatternCNN(nn.Module):
    """Compact 3-class CNN (horizontal=0, vertical=1, cross=2) for fast WSP testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 3, bias=True)
        self._init_biased_weights()

    def _init_biased_weights(self) -> None:
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.bias.zero_()
            # Channel 0: horizontal edge detector
            self.conv1.weight[0, 0] = torch.tensor([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
            # Channel 1: vertical edge detector
            self.conv1.weight[1, 0] = torch.tensor([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]])
            # Channel 2: cross intersection detector (0 on pure horiz/vert lines, +4 at cross)
            self.conv1.weight[2, 0] = torch.tensor([[0.0, 2.0, 0.0], [2.0, -4.0, 2.0], [0.0, 2.0, 0.0]])
            # Channel 3: noisy/unused channel
            self.conv1.weight[3, 0] = torch.tensor([[0.2, -0.2, 0.2], [-0.2, 0.2, -0.2], [0.2, -0.2, 0.2]])

            # Deliberately flawed fc head: class 0 bias is way too high, class 2 weight is inverted
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.5, -0.5, -0.5, 0.0],
                        [-0.5, 1.5, -0.5, 0.0],
                        [-0.5, -0.5, -1.2, 0.0],  # bug: negative weight on channel 2!
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([3.0, 0.0, -1.0]))  # bug: class 0 dominates!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def _make_cnn_dataset(n_per_class: int = 12, seed: int = 42) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    for cls in range(3):
        for i in range(n_per_class):
            img = rng.normal(0.0, 0.05, size=(1, 16, 16)).astype(np.float32)
            if cls == 0:
                # Horizontal lines
                for r in (4, 8, 12):
                    img[0, r, 2:14] += 1.0
            elif cls == 1:
                # Vertical lines
                for c in (4, 8, 12):
                    img[0, 2:14, c] += 1.0
            else:
                # Cross / plus
                img[0, 7:9, 2:14] += 1.0
                img[0, 2:14, 7:9] += 1.0
            examples.append(
                Example(
                    id=f"c{cls}_{i:02d}",
                    input=img,
                    target=cls,
                    meta={"class_name": ["horizontal", "vertical", "cross"][cls]},
                )
            )
    return examples


def _cnn_metric(pred: Any, target: Any, ex: Example) -> tuple[float, str]:
    logits = np.asarray(pred)
    pred_cls = int(np.argmax(logits))
    ok = pred_cls == int(target)
    names = ["horizontal", "vertical", "cross"]
    fb = f"pred={names[pred_cls]} target={names[int(target)]} logits={np.round(logits, 2).tolist()}"
    return (1.0 if ok else 0.0), fb


def _functiongemma_dataset() -> list[Example]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search internal company docs, HR policies, and private engineering wikis.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search public internet websites for news, weather, or general knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]
    dev_msg = {
        "role": "developer",
        "content": "You are a model that can do function calling with the following functions",
    }
    cases = [
        ("fg_01", "What is our company PTO rollover policy?", "search_knowledge_base"),
        ("fg_02", "Who won the World Cup in 2022?", "search_web"),
        ("fg_03", "Find the internal onboarding checklist for new engineers.", "search_knowledge_base"),
        ("fg_04", "What is the current population of Tokyo?", "search_web"),
        ("fg_05", "Look up our Q3 engineering architecture RFC.", "search_knowledge_base"),
        ("fg_06", "What is the stock price of NVIDIA today?", "search_web"),
        ("fg_07", "Where is the internal VPN setup guide?", "search_knowledge_base"),
        ("fg_08", "What is the boiling point of ethanol?", "search_web"),
        ("fg_09", "Check our employee expense reimbursement limit.", "search_knowledge_base"),
        ("fg_10", "Who directed the movie Inception?", "search_web"),
        ("fg_11", "Find the on-call rotation runbook for payments.", "search_knowledge_base"),
        ("fg_12", "What is the capital of Brazil?", "search_web"),
    ]
    return [
        Example(
            id=eid,
            input={"messages": [dev_msg, {"role": "user", "content": q}], "tools": tools},
            target=expected,
        )
        for eid, q, expected in cases
    ]


def _tool_metric(pred: Any, target: Any, ex: Example) -> tuple[float, str]:
    text = str(pred)
    expected = str(target)
    other = "search_web" if expected == "search_knowledge_base" else "search_knowledge_base"
    if expected in text and other not in text:
        return 1.0, f"Called {expected} correctly."
    return 0.0, f"Expected {expected}, got: {text[:160]}"


def _qwen_math_dataset() -> list[Example]:
    problems = [
        ("m_01", "What is 17 * 24? Put your final answer in \\boxed{}.", "408"),
        ("m_02", "If a train travels 60 mph for 2.5 hours, how many miles does it travel? Put your final answer in \\boxed{}.", "150"),
        ("m_03", "Solve for x: 3x + 7 = 22. Put your final answer in \\boxed{}.", "5"),
        ("m_04", "What is the area of a triangle with base 12 and height 9? Put your final answer in \\boxed{}.", "54"),
        ("m_05", "What is 15% of 200? Put your final answer in \\boxed{}.", "30"),
        ("m_06", "Compute 2^7 - 28. Put your final answer in \\boxed{}.", "100"),
        ("m_07", "A rectangle has perimeter 30 and width 6. What is its length? Put your final answer in \\boxed{}.", "9"),
        ("m_08", "What is the sum of the first 10 positive integers? Put your final answer in \\boxed{}.", "55"),
        ("m_09", "If 5 notebooks cost $15, how much do 8 notebooks cost in dollars? Put your final answer in \\boxed{}.", "24"),
        ("m_10", "What is the greatest common divisor of 36 and 48? Put your final answer in \\boxed{}.", "12"),
        ("m_11", "Evaluate (18 + 12) / 5. Put your final answer in \\boxed{}.", "6"),
        ("m_12", "What is 144 divided by 12 plus 7? Put your final answer in \\boxed{}.", "19"),
    ]
    return [
        Example(
            id=eid,
            input={"messages": [{"role": "user", "content": q}]},
            target=ans,
        )
        for eid, q, ans in problems
    ]


def _math_metric(pred: Any, target: Any, ex: Example) -> tuple[float, str]:
    text = str(pred)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    extracted = boxed[-1].strip() if boxed else ""
    ok = extracted == str(target).strip()
    return (1.0 if ok else 0.0), f"extracted={extracted!r}, target={target!r}, tail={text[-100:]!r}"


def list_builtin_tasks() -> list[dict[str, str]]:
    return [
        {
            "id": "cnn_demo",
            "name": "Small Pattern CNN (PyTorch)",
            "backend": "pytorch",
            "description": "3-class 16x16 CNN with a biased classifier head. Runs in milliseconds on CPU.",
        },
        {
            "id": "functiongemma_tools",
            "name": "FunctionGemma 270M (Tool Routing)",
            "backend": "huggingface",
            "description": "google/functiongemma-270m-it tool-selection ambiguity task (requires HF_TOKEN).",
        },
        {
            "id": "qwen_math",
            "name": "Qwen3.5-0.8B (Math Reasoning)",
            "backend": "huggingface",
            "description": "Qwen/Qwen3.5-0.8B math reasoning task from MIT Blog 1 Experiment 8.",
        },
    ]


def load_builtin_task(task_id: str, **kwargs: Any) -> tuple[StudentBackend, list[Example], MetricFn, str]:
    if task_id == "cnn_demo":
        model = SmallPatternCNN()
        backend = PyTorchBackend(model, model_id="SmallPatternCNN")
        dataset = _make_cnn_dataset()
        desc = (
            "3-class 16x16 grayscale CNN classifying patterns: 0=horizontal, 1=vertical, 2=cross. "
            "Improve classification accuracy on the blind holdout set by inspecting fc/conv weights and activations."
        )
        return backend, dataset, _cnn_metric, desc

    if task_id == "functiongemma_tools":
        model_name = kwargs.get("model_name", "google/functiongemma-270m-it")
        backend = HFBackend(model_name, max_new_tokens=96)
        dataset = _functiongemma_dataset()
        desc = (
            "FunctionGemma-270M tool-routing task: choose between `search_knowledge_base` (internal company docs) "
            "and `search_web` (public internet queries). Improve tool-selection accuracy without gradient descent."
        )
        return backend, dataset, _tool_metric, desc

    if task_id == "qwen_math":
        model_name = kwargs.get("model_name", "Qwen/Qwen3.5-0.8B")
        backend = HFBackend(model_name, max_new_tokens=256)
        dataset = _qwen_math_dataset()
        desc = (
            "Qwen3.5-0.8B math reasoning task: solve grade-school/competition math problems with final answer in \\boxed{}. "
            "Identify any looping or reasoning obstacles in the weights and fix them without gradient descent."
        )
        return backend, dataset, _math_metric, desc

    raise ValueError(f"Unknown builtin task: {task_id}")
