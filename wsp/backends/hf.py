from __future__ import annotations

import copy
import os
from typing import Any, Optional

import numpy as np
import torch

from wsp.backends.base import StudentBackend
from wsp.types import EvalResult, Example, MetricFn, Prediction, TensorInfo


class HFBackend(StudentBackend):
    """Generic HuggingFace CausalLM backend (FunctionGemma-270M, Qwen3.5-0.8B, etc.)."""

    backend_type = "huggingface"

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 128,
        hf_token: Optional[str] = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_name_or_path
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        token = hf_token or os.environ.get("HF_TOKEN")
        dtype = torch_dtype or (torch.float32 if device == "cpu" else torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=token,
            dtype=dtype,
        )
        self.base_model = base.to(self.device).eval()
        self.edited_model = copy.deepcopy(self.base_model).to(self.device).eval()

    def _model(self, edited: bool):
        return self.edited_model if edited else self.base_model

    def list_tensors(self) -> list[TensorInfo]:
        infos: list[TensorInfo] = []
        for name, param in self.edited_model.named_parameters():
            arr = param.detach().cpu().float().numpy()
            infos.append(
                TensorInfo(
                    name=name,
                    shape=list(arr.shape),
                    dtype=str(param.dtype),
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
            raise KeyError(f"Tensor '{name}' not found.")
        return params[name].detach().cpu().float().numpy().copy()

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

    def _format_input(self, raw_input: Any) -> str:
        if isinstance(raw_input, str):
            return raw_input
        if isinstance(raw_input, dict) and "messages" in raw_input:
            tools = raw_input.get("tools")
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    raw_input["messages"],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return "\n".join(f"{m['role']}: {m['content']}" for m in raw_input["messages"])
        return str(raw_input)

    def forward(
        self,
        examples: list[Example],
        metric: MetricFn,
        split: str = "trainset",
        use_edited: bool = False,
        capture_layers: Optional[list[str]] = None,
        act_dir: Optional[str] = None,
        teacher_forced: bool = False,
    ) -> EvalResult:
        model = self._model(use_edited)
        model.eval()

        captured: dict[str, list[np.ndarray]] = {}
        handles: list[Any] = []
        if capture_layers:
            modules = dict(model.named_modules())
            for layer_name in capture_layers:
                if layer_name not in modules:
                    raise KeyError(f"Layer '{layer_name}' not found.")
                captured[layer_name] = []

                def _make_hook(lname: str):
                    def _hook(_mod, _inp, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        if isinstance(out, torch.Tensor):
                            # Save last-token or full hidden state as float32 numpy
                            captured[lname].append(out.detach().cpu().float().numpy())
                    return _hook

                handles.append(modules[layer_name].register_forward_hook(_make_hook(layer_name)))

        predictions: list[Prediction] = []
        total_score = 0.0
        try:
            with torch.no_grad():
                for ex in examples:
                    prompt_text = self._format_input(ex.input)
                    inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                    if teacher_forced:
                        out = model(**inputs)
                        last_logits = out.logits[0, -1].detach().cpu().float().numpy()
                        top_id = int(np.argmax(last_logits))
                        text_out = self.tokenizer.decode([top_id])
                    else:
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                        new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
                        text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

                    score, feedback = metric(text_out, ex.target, ex)
                    total_score += float(score)
                    predictions.append(
                        Prediction(
                            example_id=ex.id,
                            output=text_out,
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
                # Pad or take last token if sequence lengths differ
                if len(chunks) == 1:
                    arr = chunks[0]
                else:
                    arr = np.stack([c[0, -1] if c.ndim == 3 else c[0] for c in chunks], axis=0)
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
