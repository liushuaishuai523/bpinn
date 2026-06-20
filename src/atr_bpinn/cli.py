from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-threshold ATR-BPINN and extrapolate gamma to infinity."
    )
    parser.add_argument("--profile", choices=("quick", "paper"), default="paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse converged runs in the output folder")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .experiment import run_experiment

    if args.profile == "quick":
        config = ExperimentConfig.quick(args.output or Path("results/quick"))
    else:
        config = ExperimentConfig(output_dir=args.output or Path("results/paper"))
    summary = run_experiment(config, args.device, resume=args.resume)
    combined = summary.get("combined")
    if not combined or not combined.get("valid", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
