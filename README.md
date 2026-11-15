# WSP Studio — General-Purpose Weight-Space Programming

A modular, model-agnostic Weight-Space Programming (WSP) harness inspired by MIT CSAIL's *Sparks of Weight-Space Programming* and Cmpnd's `dspy-wsp`.

> 📖 **Concept & Deep Dive:** See [docs/weight_space_programming_explained.md](docs/weight_space_programming_explained.md) for a step-by-step explanation of Weight-Space Programming, the MIT CSAIL research, and `dspy-wsp`.

## Features
- **Model-Agnostic Backends (`wsp/backends/`):**
  - `PyTorchBackend`: Any `torch.nn.Module` (ResNets, CNNs, MLPs) with generic forward-hook activation capture.
  - `HFBackend`: Any HuggingFace `AutoModelForCausalLM` (e.g., `google/functiongemma-270m-it`, `Qwen/Qwen3.5-0.8B`).
  - `NpzBackend`: Lightweight NumPy MLP/policy networks.
- **Writer LLM Providers (`wsp/writers/`):**
  - `GeminiWriter`: Uses `GEMINI_API_KEY` (`gemini-2.5-pro`, `gemini-2.5-flash`, etc.).
  - `BedrockWriter`: Uses `boto3` AWS Bedrock Runtime `converse` API.
- **Data Hygiene & Patching:** Automatic `trainset` / blind `holdout` / hidden `valset` splits, isolated Python sandbox, and SHA-256-verified `manifest.json` + `edit.npz` patch export.

## Quick Start

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 1. Run via CLI
With **Gemini**:
```bash
export GEMINI_API_KEY="your-key"
python cli.py --task cnn_demo --writer gemini --model gemini-2.5-pro --max-passes 15
```

With **AWS Bedrock**:
```bash
export AWS_REGION="us-east-1"
python cli.py --task cnn_demo --writer bedrock --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --max-passes 15
```

With **FunctionGemma 270M** (requires HF license acceptance + `HF_TOKEN`):
```bash
export HF_TOKEN="hf_..."
python cli.py --task functiongemma_tools --writer gemini --max-passes 15
```

### 2. Run with Your Own Small CNN
Create a file `my_cnn_task.py` defining `load_task()`:
```python
from wsp.backends.pytorch import PyTorchBackend
from wsp.types import Example

def load_task():
    model = ... # your nn.Module loaded from .pt/.pth
    backend = PyTorchBackend(model, model_id="my-resnet")
    dataset = [Example(id="img_0", input=tensor_or_np, target=label), ...]
    def metric(pred, target, ex):
        ok = int(pred.argmax()) == int(target)
        return (1.0 if ok else 0.0), f"pred={int(pred.argmax())} target={target}"
    return backend, dataset, metric, "Improve accuracy on my CNN."
```
Then run:
```bash
python cli.py --task-file my_cnn_task.py --writer gemini --max-passes 20
```

### 3. Launch the Web UI
```bash
uvicorn server:app --host 0.0.0.0 --port 8765
```

## Running Tests
```bash
pytest -v
```

