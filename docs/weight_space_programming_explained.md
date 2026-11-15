# Weight-Space Programming (WSP): Concept, Mechanics, and Step-by-Step Breakdown

This document explains **Weight-Space Programming (WSP)**—what it is, how it works in the original MIT CSAIL research (*Sparks of Weight-Space Programming*), and how Cmpnd productized it into a reusable optimizer in `dspy-wsp`.

---

## 1. What Is Weight-Space Programming?

Traditionally, there have been two ways to change how a neural network behaves:
1. **Prompt / Harness Optimization** (e.g., DSPy `MIPROv2`, `GEPA`): An LLM edits *text* (instructions, few-shot examples, tool harnesses). The model's weights never change.
2. **Gradient-Based Training / Fine-Tuning** (e.g., SGD, Adam, PPO, `BootstrapFinetune`): An optimizer computes a loss and uses backpropagation/gradients over a dataset to nudge millions or billions of weights.

**Weight-Space Programming (WSP)** introduces a third paradigm: **treating a neural network's weights like source code**—something a frontier "writer" LLM can *read*, *reason about*, and *directly edit* in natural language and Python, **without gradient descent or a training loop**.

Instead of a blind numerical optimizer taking millions of tiny gradient steps, a frontier reasoning model:
1. Inspects the student model's weight matrices (and optionally its internal activations on a few examples).
2. Forms a mechanistic hypothesis (e.g., *"neurons 2 and 3 in layer 3 over-fire on the word 'invoice'"* or *"layers 15 and 19 write a direction that causes reasoning loops"*).
3. Writes a targeted numerical patch to a tiny fraction of the weights.
4. Verifies the change with a small, fixed budget of forward passes.

### Key Mental Model: What WSP Can and Cannot Do
- **Where it shines:** Removing obstacles that mask an existing capability (e.g., fixing a looping circuit, breaking a local-minimum policy, recalibrating a decision head) or hand-crafting small, structured policy networks from scratch.
- **Where it struggles:** Teaching a model brand-new knowledge or capabilities it fundamentally lacks. For example, in the MIT study, Qwen3.5-0.8B jumped from **7.9% → 48.1%** on MATH-500 because the edit stopped it from looping; on Gemma 3 270M (which didn't loop and simply lacked the math capability), WSP produced no meaningful gain.

---

## 2. How It Works in Blog 1: *Sparks of Weight-Space Programming* (MIT CSAIL)

*Source: [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab — Nov 2, 2026)*

The MIT team tested a frontier reasoning model (**GPT-6 Astra**) across three modes: **Reading**, **Writing from Scratch**, and **Fixing/Improving** existing networks.

### Mode A: Reading Weights (Interpretability Without Running the Model)

#### Experiment 1 — Identifying Atari/Grid Games (No Tools)
- **Setup:** Four PPO policy MLPs (`500 → 32 → 32 → 5`, `tanh`, 17,253 weights each) trained on Breakout, Pong, Pac-Man, and Freeway. All four share the exact same 10×10 5-channel grid input (`agent`, `moving objects`, `reward targets`, `obstacles`, `previous frame`) and 5 action logits (`no-op, up, down, left, right`).
- **Step-by-step:**
  1. All 17,253 raw weights are serialized as text into the prompt with no game name or options given.
  2. Astra scans the first-layer weights (`W1`) by input channel and grid coordinate. For Breakout, it notices large weights concentrated along **row 8** of Channel 0 (a horizontal paddle at the bottom) and **rows 1–3** of Channel 2 (bricks).
  3. It checks the output layer (`W3`) and sees `left`/`right` logits strongly differentiated while `no-op`/`up`/`down` are grouped.
  4. **Result:** Astra correctly names all 4 games from raw weights alone.

#### Experiment 2 — Identifying Diffusion Shapes (No Tools, No Forward Passes)
- **Setup:** Three DDPM MLPs (`160 → 64 → 64 → 144`, ReLU, 23,824 weights) generating 12×12 grayscale circles, squares, or triangles.
- **Step-by-step:**
  1. All 23,824 weights are passed as text; forward passes are forbidden.
  2. Astra inspects the output layer weights (`W3`), mapping each hidden unit's 144 output weights back to a 12×12 spatial grid.
  3. It recognizes localized positive/negative edge bands (e.g., curved rim segments forming a circular outline).
  4. **Result:** Correctly identifies circle, square, and triangle (notably drawing them as outlines rather than filled shapes, proving it was reading weight geometry rather than mentally simulating forward passes).

#### Experiment 3 — Reading Middle-Layer Transformers (Python Kernel, No Forward Passes)
- **Setup:** Three 4-layer, 6-head character transformers (7.16M weights) trained on Python code, Japanese romaji, or English Wikipedia. **The input embedding and output unembedding layers are removed**—Astra only gets the 4 middle transformer blocks and a Python kernel (up to 100 runs), with running the network on *any* input strictly forbidden.
- **Step-by-step:**
  1. Without embeddings, Astra cannot read characters directly. Instead, it writes Python scripts to multiply `W_Q`, `W_K`, `W_V`, and `W_O` matrices across layers (independently rediscovering Elhage et al.'s Transformer Circuits composition math).
  2. It identifies **previous-token heads** in layer 0 and **induction (copying) heads** in layers 2–3 that compose with them.
  3. Because code repeats variable names/identifiers heavily, the Python model has 3 copying heads (`2.1, 3.1, 3.5`), which Astra finds with 100% precision. For romaji, it observes that the final block writes into fewer distinct subspaces (fewer active characters, since `q, l, v, x` don't appear in romaji).

---

### Mode B: Writing Weights from Scratch (Zero-Shot, No Training)

#### Experiment 5 — MinAtar Policies Without Data or Gradients
- **Setup:** `600 → 32 → 32 → 6` (`tanh`) policy networks for MinAtar Breakout, Seaquest, and Asterix, initialized to all zeros. Astra is given the game rules, source code, and I/O layout—no tools, no rollouts, no gradients.
- **Step-by-step:**
  1. Astra designs a hand-coded neural algorithm in its reasoning trace (e.g., for Breakout: *"Units 0..3: descending-right, descending-left... Subtract paddle position from ball position..."*).
  2. It emits the exact floating-point weight matrices as text in a single reply.
  3. **Result:** Evaluated over 1,000 episodes, Astra's zero-shot weights beat 100M-step PPO on all three games, and on Breakout score **517.4** vs. 500M-step PPO's **25.3**.

---

### Mode C: Fixing and Improving Existing Models

#### Experiment 4 — Fixing a Stuck RL Policy (1-Shot, No Tools)
- **Setup:** A 171,664-weight Kinetix PPO actor (`679 → 128 → 5×128 → 16`) that drives straight into a ramp and stalls (0% success across 256 episodes).
- **Step-by-step:**
  1. Astra receives the weights, scene description, and 1 failed episode trace.
  2. It diagnoses a local minimum: the car needs to reverse first to build runway speed.
  3. It writes a single reply modifying **3.5% of the weights** to carve out a small bypass circuit through a few units per layer—reading car position/speed to reverse, then switching off once past the red strip so the original trained policy finishes the jump.
  4. **Result:** **100% success** across 1,024 test episodes.

#### Experiment 8 — Improving `Qwen3.5-0.8B` Math Performance (Python Kernel + 50 Forward Passes)
This is the flagship real-LLM experiment:
1. **Problem:** Base `Qwen3.5-0.8B` scores only **7.9%** on MATH-500 because it falls into repetitive thinking loops and hits the 32,768-token limit on **93.0%** of problems.
2. **Environment:** Astra is given a Python session with access to the model's weights and a strict budget of **50 forward passes** on practice problems from GSM8K/MATH training sets (no gradients, no training loop).
3. **Diagnosis & Localization:** Using Python to inspect activations and weight projections during a few forward passes, Astra traces the runaway repetition loop to a **single direction** written into the residual stream by the MLPs in **layers 15 and 19**.
4. **Surgical Edit:** Astra subtracts that direction from **64 output columns of `down_proj`** in layers 15 and 19—modifying **131,072 weights** (**0.016%** of the 0.8B model).
5. **Result (same sampling settings, no further training):**
   - **MATH-500:** `7.9%` → **48.1%** (token-limit hits drop `93.0%` → `33.1%`)
   - **GSM8K:** `13.8%` → **67.1%**
   - **IFEval (instruction following):** `21.0%` → **62.3%** (generalizes out-of-the-box)

---

## 3. How It Works in Blog 2: *Cmpnd Releases `dspy-wsp`*

*Sources: [Cmpnd Blog: Weight-Space Programming with DSPy](https://www.cmpnd.ai/blog/weight-space-programming-dspy.html) (Nov 13, 2026) & [MarkTechPost: Cmpnd Releases dspy-wsp](https://www.marktechpost.com/2026/11/14/cmpnd-dspy-wsp-weight-space-programming/?amp=1) (Nov 14, 2026)*

Cmpnd took the MIT research loop and packaged it as **`WeightSpace`**, a drop-in **DSPy optimizer** (`pip install dspy-wsp`) that works with any DSPy `Signature`, `Module`, and `metric`.

### The 3-Actor Architecture

1. **The Student:** A small local open-weight model (`0.6B` to `9B`, e.g., `d1-omni-600M`, `NanoJev`, `d1-2B`, `Clef-Flash 9B`, or text LLMs) loaded into a local Python process (MLX on Apple Silicon; PyTorch/CUDA or vLLM on Linux).
2. **The Writer:** A frontier reasoning LLM (recommended: `gpt-6-astra` with `reasoning_effort="max"`).
3. **The Sandbox:** An isolated local execution environment (`sandbox-exec` on macOS, `bubblewrap` on Linux) with **no GPU, no network, and no gradients**. It holds the student's weights, scratch notes (`notes.md`), captured activations (`fwd_*/mlp_act.npy`), and candidate edits (`edited/`).

### Strict Data Separation (Preventing Overfitting)
- **`trainset`:** Practice examples the writer can run, inspect outputs on, and capture activations from.
- **`holdout` (e.g., `0.25`):** A blind check split carved out of `trainset`. The writer can *score* its edits on `holdout` to verify generalization, but never sees the raw examples.
- **`valset`:** Completely hidden until optimization finishes; used once at the end to compare the original vs. edited model on identical examples and seeds.

### Step-by-Step Execution Loop (`WeightSpace.compile`)

The writer operates in **turns** against a fixed forward-pass budget (e.g., `max_passes=25`). Every turn contains a short natural-language **REASON** (kept in the transcript) followed by **one ACTION block**—either a `forward` block or a `python` block:

1. **Step 1 — Brief the Writer:**
   The harness initializes the sandbox with the student's weights and briefs the writer on the DSPy task signature, metric, tensor names/shapes, and remaining forward-pass budget.

2. **Step 2 — Run Baseline (`forward` block):**
   On Turn 1, the writer runs the unedited student on a small batch of practice examples:
   ```json
   {"type": "generate", "model": "original", "problems": ["t01", "...", "t08"]}
   ```
   The harness executes the DSPy program, computes the `metric` (which can return `dspy.Prediction(score=..., feedback=...)`), and writes results (`gen_1.jsonl`) for the writer to read.

3. **Step 3 — Look Inside (`forward` with activation capture):**
   Seeing which examples failed (e.g., billing tickets misrouted), the writer requests an internal trace on a failing example:
   ```json
   {"type": "teacher_forced", "model": "original", "texts": ["<ticket t04>"], "return": {"mlp_act": [3]}}
   ```
   The harness runs a teacher-forced pass and saves the requested layer activations to `fwd_2/mlp_act.npy`.

4. **Step 4 — Hypothesize & Edit (`python` block):**
   The writer writes Python to inspect the `.npy` activations, find which neurons/columns or low-rank directions are responsible, and patch the weights:
   ```python
   acts = np.load("fwd_2/mlp_act.npy")
   w = load("blocks.3.mlp.down")
   w[:, [2, 3]] *= 0.4
   save_edit("blocks.3.mlp.down", w)
   ```

5. **Step 5 — Test & Iterate (`forward` on `"both"`):**
   The writer tests the candidate edit side-by-side against the original model on the same examples and seeds:
   ```json
   {"type": "generate", "model": "both", "problems": ["t01", "...", "t08"]}
   ```
   This loop repeats until `max_passes` is exhausted.

6. **Step 6 — Export Verifiable Patch:**
   The best-scoring edit is saved to `run_dir` as a compact, reproducible patch:
   - `manifest.json` (with SHA-256 hashes of every base and edited tensor)
   - `edit.npz` (containing *only* the modified entries—e.g., 1,024 entries in `NanoJev` or a rank-9 diff in `d1-2B`)
   Anyone with the public base model can deterministically reconstruct the edited checkpoint (`compiled.wsp_patch.write_checkpoint(...)`).

### Benchmark Results & Practical Realities (`Typed Decisions`, 400 test cases)
| Model | Unoptimized | ReAnchor | WeightSpace | WeightSpace + ReAnchor | Writer Cost / Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **d1-omni-600M** | 50.9% | 60.2% (+9.3) | 66.9% (+16.1) | **69.1% (+18.3)** | \$7.63 / 78 min |
| **NanoJev (Qwen3-0.6B)** | 39.2% | 50.7% (+11.6) | 52.4% (+13.3) | **56.4% (+17.3)** | \$6.92 / 167 min |
| **d1-2B** | 60.6% | 70.1% (+9.5) | 74.0% (+13.4) | **77.9% (+17.3)** | \$5.42 / 83 min |
| **Clef-Flash (9B)** | 68.8% | 70.3% (+1.5) | 70.7% (+1.9) | **75.3% (+6.5)** | — |

**Key Takeaways from Cmpnd's Findings:**
- **Composes with threshold calibration:** `WeightSpace` edits internal representations, while `ReAnchor` recalibrates output decision thresholds. Combining them produced the best scores on every model.
- **Writer strength matters:** Switching from `gpt-6-astra` to `gpt-6.1-sol` dropped `d1-omni-600M` gains from `+16.1` to `+5.8`, and `d1-2B` from `+13.4` to `-0.6`.
- **Scaling & rules:** Gains shrink on larger models (`Clef-Flash 9B` only gained `+1.9` alone). Also, `dspy-wsp` allows the writer to recompute the decision head inside the Python kernel by default; disabling that (`simulate_head=False`, matching the stricter MIT rule) dropped `d1-omni-600M` from `+16.1` to `+2.8`.

---

## 4. Why Your Current App Doesn't Do This Yet (and How to Build One)

Right now, your [app.py](file:///home/antigravity/upload/app.py) is a **read-only weight inspector** for Experiment 1's `.npz` files—it loads weight matrices and sends summaries to an LLM, which is why it can't actually modify or improve a model.

To build a working **Weight-Space Programming App** (like a mini `dspy-wsp`), you need four components wired into a loop:
1. **Student Model Runner & Activation Hook:** Loads a small local model (e.g., PyTorch/Transformers or an MLP/RL policy), runs evaluation on a `trainset`/`holdout`, and can dump intermediate activations (`.npy`) on demand.
2. **Sandboxed Python Kernel:** A restricted Python execution environment where the writer LLM can load weight tensors, inspect activation `.npy` files, compute SVD/projections, and write modified tensors to an `edited/` directory.
3. **Writer LLM Turn Loop (`max_passes` budget):** Prompts a strong reasoning LLM with the task, metric feedback, and available actions (`forward` or `python`), parsing one action block per turn until the budget is spent.
4. **Patch & Verification Manager:** Saves the best edit as an `edit.npz` + `manifest.json` diff, evaluates it once on the hidden `valset` against the original model, and lets you export or serve the patched checkpoint.
