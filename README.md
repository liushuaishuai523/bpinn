# BPINN Blow-Up PDE Experiments

This repository contains compact experiment scripts for physics-informed neural network methods on PDEs with blow-up behavior. Directory names describe the PDE example being solved, and script names describe the method family used in that example.

Generated results, checkpoints, figures, IDE files, and cache files are ignored by `.gitignore`.

## Project Structure

| Path | Example |
| --- | --- |
| `semilinear_parabolic_example_4_1/` | Semi-linear parabolic Example 4.1 with blow-up time near `0.1664`. |
| `semilinear_parabolic_example_2/` | Semi-linear parabolic Example 2 with blow-up time near `0.026543`. |
| `convection_diffusion_example_6_4/` | Convection-diffusion blow-up Example 6.4 with blow-up time near `0.16665`. |
| `fourth_order_blowup/` | Fourth-order blow-up equation with blow-up time near `0.9598`. |

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version before running long experiments.

## Running Experiments

Run each script from its own example folder because outputs are written with relative paths:

```bash
cd semilinear_parabolic_example_4_1
python bpinn_fi_bpinn.py --filename demo_bpinn --use_fi 0
```

Each script writes generated artifacts to:

| Path | Content |
| --- | --- |
| `results/{filename}/` | Loss curves, error curves, solution plots, and NumPy data. |
| `checkpoints/{filename}_final_model.pth` | Final trained model weights. |

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--e` | Number of training epochs. |
| `--number` | Sampling counts. |
| `--ranges` | Spatial and temporal domain ranges. |
| `--filename` | Output folder and checkpoint filename prefix. |
| `--use_fi` | Enable failure-informed adaptive sampling. |
| `--use_apinn` | Enable adaptive loss weighting, where available. |
| `--use_time_loss` | Enable additional time-grouped loss, where available. |

## Script Index

### Semi-Linear Parabolic Example 4.1

| Script | Purpose |
| --- | --- |
| `semilinear_parabolic_example_4_1/bpinn_fi_bpinn.py` | BPINN and FI-BPINN experiments. |
| `semilinear_parabolic_example_4_1/adaptive_bpinn.py` | A-BPINN experiment with adaptive weighting options. |
| `semilinear_parabolic_example_4_1/pinn_fi_pinn_apinn.py` | PINN, FI-PINN, and A-PINN experiments. |

### Semi-Linear Parabolic Example 2

| Script | Purpose |
| --- | --- |
| `semilinear_parabolic_example_2/bpinn_fi_bpinn.py` | BPINN and FI-BPINN experiments. |
| `semilinear_parabolic_example_2/adaptive_bpinn.py` | A-BPINN experiment. |
| `semilinear_parabolic_example_2/pinn_fi_pinn_apinn.py` | PINN, FI-PINN, and A-PINN experiments. |

### Convection-Diffusion Example 6.4

| Script | Purpose |
| --- | --- |
| `convection_diffusion_example_6_4/bpinn_fi_bpinn.py` | BPINN and FI-BPINN experiments. |
| `convection_diffusion_example_6_4/adaptive_bpinn.py` | A-BPINN experiment. |
| `convection_diffusion_example_6_4/pinn_fi_pinn_apinn.py` | PINN, FI-PINN, and A-PINN experiments. |
| `convection_diffusion_example_6_4/atr_bpinn_blowup_time.py` | ATR-BPINN blow-up time prediction. |

### Fourth-Order Blow-Up Equation

| Script | Purpose |
| --- | --- |
| `fourth_order_blowup/bpinn_fi_bpinn.py` | BPINN and FI-BPINN experiments. |
| `fourth_order_blowup/adaptive_bpinn.py` | A-BPINN experiment. |
| `fourth_order_blowup/pinn_fi_pinn_apinn.py` | PINN, FI-PINN, and A-PINN experiments. |
| `fourth_order_blowup/atr_bpinn_blowup_time.py` | ATR-BPINN blow-up time prediction. |
| `fourth_order_blowup/plot_3d_fi_bpinn.py` | 3D visualization variant for fourth-order results. |

## Data and Checkpoints

Do not commit generated folders such as `results/`, `checkpoints/`, `model/`, `*_result*/`, `figure/`, or `compare_data/`. Re-run the scripts to regenerate results locally. If trained models or selected final figures need to be shared, place them in GitHub Releases or an external data repository and link them here.

## License

No license has been selected yet. Add a `LICENSE` file before making the repository public if others should be allowed to reuse the code.
