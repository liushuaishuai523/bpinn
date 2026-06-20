# BPINN and ATR-BPINN

Reproducible PyTorch implementation of the Blow-up Physics-Informed Neural Network (BPINN) and Adaptive Time Refinement BPINN (ATR-BPINN) experiments. The repository contains four fixed-domain BPINN examples, fourth-order ATR-BPINN prediction, reference solvers, manuscript figure generation, saved
checkpoints, and machine-readable result files.

## Features

- Four blow-up PDE examples implemented through a shared BPINN interface.
- Strictly positive transformed variable with log-scale geometric coupling.
- Exact initial and Dirichlet constraints.
- Navier boundary enforcement for the fourth-order equation.
- Characteristic-coordinate input for the convection-diffusion equation.
- Fixed-threshold ATR-BPINN with linear and quadratic $1/\gamma$ diagnostics.
- Reproducible seeds, JSON/CSV outputs, PDF figures, tests, and GitHub Actions.

## Repository Layout

```text
.
├── run_bpinn_examples.py          # Run the four fixed-domain BPINN examples
├── run_experiment.py              # Run fourth-order ATR-BPINN
├── generate_manuscript_figures.py # Export manuscript-ready PDF figures
├── src/atr_bpinn/                 # Models, problems, training, and analysis
├── tests/                         # Unit tests
├── results/bpinn/                 # BPINN metrics, figures, and checkpoints
├── results/quick/                 # Verified ATR-BPINN threshold matrix
├── requirements.txt
└── pyproject.toml
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer
- NumPy, SciPy, and Matplotlib

Double precision is used for the fourth derivatives. With `--device auto`,
CUDA is selected when available; otherwise the experiments run on CPU.

## Installation

```bash
git clone https://github.com/liushuaishuai523/bpinn.git
cd bpinn

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run BPINN Examples

Run the FK example with the publication configuration:

```bash
python run_bpinn_examples.py \
  --problem fk \
  --profile paper \
  --device auto \
  --output results/bpinn_fk
```

Run all four examples:

```bash
python run_bpinn_examples.py \
  --problem all \
  --profile paper \
  --device auto \
  --output results/bpinn
```

Available problem names are:

| Name | Equation | Evaluation time |
|---|---|---:|
| `fk` | $u_t=u_{xx}+3\exp(u)$ | 0.1663 |
| `cubic` | $u_t=u_{xx}+5(u+1)^3$ | 0.0264 |
| `fourth` | $u_t=-u_{xxxx}+\exp(u)$ | 0.9598 |
| `convection` | $u_t+u_x=u_{xx}+3\exp(u)$ | 0.16664 |

Use `--profile quick` for a short pipeline check. The `paper` profile uses the
validated network sizes, sample counts, and optimization budgets.

Each run writes the following files to the selected output directory:

- `<problem>_model.pt`: trained PyTorch checkpoint;
- `<problem>_bpinn.png` and `.pdf`: solution and error figure;
- `metrics.csv`: global and terminal relative errors;
- `summary.json`: configuration, losses, and quality-gate results.

## Run ATR-BPINN

Run the verified threshold matrix:

```bash
python run_experiment.py \
  --profile quick \
  --device auto \
  --output results/atr_quick
```

Run the publication configuration:

```bash
python run_experiment.py \
  --profile paper \
  --device auto \
  --output results/atr_paper
```

Continue an interrupted matrix without repeating converged runs:

```bash
python run_experiment.py \
  --profile paper \
  --device auto \
  --output results/atr_paper \
  --resume
```

ATR-BPINN evaluates the common threshold set
`{50, 100, 200, 400, 800}` for initial terminal times `{1.2, 1.5, 1.8}`.
The output directory contains all refinement records, threshold predictions,
fit diagnostics, the combined estimate, and PNG/PDF figures.

## Generate Manuscript Figures

After running the four BPINN examples with output `results/bpinn`, generate the
manuscript-ready PDF figures with:

```bash
python generate_manuscript_figures.py \
  --results results/bpinn \
  --atr-results results/quick \
  --output results/manuscript_figures \
  --device auto
```

The exported PDFs use embedded fonts, enlarged labels, and 300 dpi rasterized
heatmaps.

## Tests

```bash
python -m pytest -q
```

## Results

Existing verified outputs are included under `results/bpinn/` and `results/quick/`.

## Reproducibility

- Default random seed: `20260620`.
- Training and evaluation use `float64` unless explicitly changed.
- Initial and boundary constraints are embedded in the fixed-domain models.
- Commands return a nonzero exit code when an accuracy or convergence gate
  fails.
- Full configurations can require several minutes on CPU, especially for the
  fourth-order and convection-diffusion examples.
