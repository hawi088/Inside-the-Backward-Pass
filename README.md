# Gradienr – From Hand‑Calculated Forward Pass to a Tiny NumPy Engine

## Overview
Gradient is a teaching‑oriented project that walks through the full life‑cycle of a minimal neural network built from pure NumPy. It demonstrates how a hand‑calculated forward and backward pass (Phase 1) can be reproduced exactly by a small, modular engine (Phase 2) and provides a suite of visualisations and benchmarks.

## Project Structure
```
gradient/
│
├─ engine/                     # Core engine package
│   ├─ __init__.py
│   ├─ layers.py               # Linear, Sigmoid, ReLU, Tanh
│   ├─ losses.py               # MSELoss, BCELoss
│   ├─ network.py              # Sequential container
│   ├─ optimizers.py           # SGD, SGDMomentum, Adam
│   └─ grad_check.py           # Finite‑difference gradient checker
│
├─ dataset/
│   ├─ __init__.py
│   └─ two_moons.py            # Simple 2‑D synthetic dataset generator
│
├─ exports/                    # Generated figures (git‑ignored)
│
├─ activation_benchmark.py     # Plots activation functions and derivatives
├─ derivative_visualizer.py    # Visualises numerical vs analytical derivatives
├─ engine.py                   # Driver for the modular engine (Phase 2)
├─ generate_3b1b_suite.py      # Generates the 3‑b‑1‑b illustration suite
├─ gradient_checker_proof.py   # Runs the rigorous gradient‑check proof (Phase 3)
├─ interactive_explorer.py     # CLI helper to run individual scripts
├─ linear_model.py             # Simple linear regression example
├─ loss_landscape.py           # 2‑D loss surface visualisation for the tiny net
├─ optimizer_benchmark.py      # Benchmarks SGD, SGDMomentum, Adam on the tiny net
├─ hand_forward_compare.py    # Phase 1 reference calculation (hand‑pencil) 
├─ README.md                  # This document
└─ requirements.txt            # Python dependencies
```

## Setup
1. **Python** – 3.11 or newer (tested on Windows 10/11).
2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   `requirements.txt` includes `numpy>=1.26` and `matplotlib>=3.8`.

## Running the Project
### Phase 1 – Hand‑Calculated Verification
```bash
python hand_forward_compare.py
```
Prints the input vector, hidden pre‑activations, activations, output, loss, and all gradients. Numbers are computed with `float64` and match the values shown in the accompanying paper.

### Phase 2 – Modular Engine
```bash
python engine.py            # prints trace and a batch summary
python engine.py --show     # also displays a Matplotlib inspection plot
```
Key functions in `engine.py`:
* `create_phase2_network()` – builds a `Sequential` model with Xavier‑initialized weights.
* `run_phase1_paper_trace_check(model)` – injects the exact Phase 1 weights, runs forward and backward passes, and prints every intermediate value for direct comparison.
* `plot_module_3_inspection(...)` – saves `exports/module3_engine_summary.png` visualising weight matrices and gradient norms.

### Additional Scripts
| Script | Purpose |
|--------|---------|
| `activation_benchmark.py` | Visualises Sigmoid, ReLU, Tanh functions and their derivatives. |
| `derivative_visualizer.py` | Compares numerical finite‑difference derivatives with analytical ones for each activation. |
| `optimizer_benchmark.py` | Trains the tiny network with SGD, SGDMomentum, and Adam; plots loss curves and final parameters. |
| `loss_landscape.py` | Sweeps the two output‑layer weights while keeping other parameters fixed, producing a 2‑D loss surface heatmap. |
| `gradient_checker_proof.py` | Executes a symmetric finite‑difference gradient check for all nine parameters; prints a proof verdict and saves `exports/module7_gradient_checker_proof.png`. |
| `interactive_explorer.py` | Small command‑line helper to select a script, enable `--show`, or change the random seed. |
| `linear_model.py` | Demonstrates a basic linear regression example using the engine. |
| `generate_3b1b_suite.py` | Generates the animation suite used for the project’s supplementary videos. |

All visual outputs are stored under `exports/` and share a dark‑background style for consistency.


## Guided Execution Flow

The project is organized into a series of scripts that illustrate different concepts. Follow this order to reproduce the full story:

1. **Phase 2 – Gradient checker**
   ```bash
   python checker.py  # (if you have a helper script; otherwise run the engine verification step)
   ```
   *Validates that the analytical gradients of the modular engine match the finite‑difference baseline.*

2. **Gradient‑check proof** – `gradient_checker_proof.py`
   ```bash
   python gradient_checker_proof.py
   ```
   *Runs a rigorous symmetric finite‑difference check for every parameter and prints a verdict.*

3. **Derivative visualizer** – `derivative_visualizer.py`
   ```bash
   python derivative_visualizer.py
   ```
   *Shows intuitive numerical vs. analytical derivatives for each activation function.*

4. **3‑b‑1‑b illustration suite** – `generate_3b1b_suite.py`
   ```bash
   python generate_3b1b_suite.py
   ```
   *Creates the animation suite that ties the forward‑pass story together.*

5. **Linear model failure demo** – `linear_model.py`
   ```bash
   python linear_model.py
   ```
   *Demonstrates why a simple linear model fails to classify the two‑moons dataset.*

6. **Loss landscape** – `loss_landscape.py`
   ```bash
   python loss_landscape.py
   ```
   *Visualises the 2‑D loss surface of the tiny network by sweeping the output‑layer weights.*

7. **Activation dynamics** – `module5_activation_dynamics.py`
   ```bash
   python module5_activation_dynamics.py
   ```
   *Shows how different activation functions affect the forward‑pass output and gradients.*

8. **Optimizer benchmark** – `optimizer_benchmark.py`
   ```bash
   python optimizer_benchmark.py
   ```
   *Compares SGD, SGD + Momentum, and Adam on the same tiny network.*

## License
This project is released under the MIT License. See `LICENSE` for details.