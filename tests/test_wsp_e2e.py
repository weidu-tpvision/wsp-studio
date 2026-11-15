from __future__ import annotations

import os
import tempfile

from wsp.backends.pytorch import PyTorchBackend
from wsp.optimizer import WSPOptimizer
from wsp.tasks.builtin import SmallPatternCNN, load_builtin_task
from wsp.writers.mock import MockWriter


def test_e2e_pytorch_cnn_wsp_and_patch_roundtrip():
    backend, dataset, metric, desc = load_builtin_task("cnn_demo")

    script = [
        # Turn 1: Baseline forward + capture penultimate pool activations
        (
            'REASON: Run baseline on trainset and capture pool layer activations.\n'
            '```forward\n'
            '{"model": "original", "split": "trainset", "capture_layers": ["pool"]}\n'
            '```'
        ),
        # Turn 2: Inspect activations and fix fc.weight + fc.bias without gradients
        (
            'REASON: Inspect pool activations and fix inverted channel 2 weight and biased class 0 bias.\n'
            '```python\n'
            'acts = load_act("fwd_1/pool.npy")\n'
            'print("Captured pool shape:", acts.shape)\n'
            'w = load_weight("fc.weight")\n'
            'b = load_weight("fc.bias")\n'
            'w[2, 2] = 2.0\n'
            'b[:] = [0.0, 0.0, 0.0]\n'
            'print(save_edit("fc.weight", w))\n'
            'print(save_edit("fc.bias", b))\n'
            '```'
        ),
        # Turn 3: Verify edited vs original
        (
            'REASON: Compare edited vs original on trainset and trigger blind holdout scoring.\n'
            '```forward\n'
            '{"model": "both", "split": "trainset"}\n'
            '```'
        ),
        # Turn 4: Finish
        (
            'REASON: Accuracy reached 100% on holdout.\n'
            '```finish\n'
            '{"summary": "Fixed fc.weight[2,2] and fc.bias."}\n'
            '```'
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        opt = WSPOptimizer(
            backend=backend,
            writer=MockWriter(script=script),
            metric=metric,
            task_description=desc,
            max_passes=10,
            run_dir=tmpdir,
        )
        res = opt.compile(dataset)

        assert res.val_score_before < 0.7
        assert res.val_score_after == 1.0
        assert res.holdout_score_after == 1.0
        assert res.manifest.modified_params > 0
        assert "fc.weight" in res.manifest.modified_tensors
        assert "fc.bias" in res.manifest.modified_tensors

        patch_dir = os.path.join(tmpdir, "patch")
        assert os.path.exists(os.path.join(patch_dir, "manifest.json"))
        assert os.path.exists(os.path.join(patch_dir, "edit.npz"))

        # Verify patch can be applied deterministically to a fresh model instance
        fresh_backend = PyTorchBackend(SmallPatternCNN(), model_id="SmallPatternCNN")
        fresh_backend.apply_patch(patch_dir, verify_hashes=True)
        reloaded_eval = fresh_backend.forward(dataset, metric, split="valset", use_edited=True)
        assert reloaded_eval.score == 1.0


def test_sandbox_blocks_gradient_training():
    from wsp.sandbox import SandboxedKernel

    backend, _, _, _ = load_builtin_task("cnn_demo")
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel = SandboxedKernel(backend, tmpdir)
        out = kernel.run_python("loss.backward()\n")
        assert "forbidden" in out.lower()
