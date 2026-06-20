from __future__ import annotations

import argparse
from pathlib import Path

from .problems import PROBLEMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-domain BPINN on the four manuscript examples."
    )
    parser.add_argument("--profile", choices=("quick", "paper"), default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--problem", choices=("all", *PROBLEMS), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/bpinn"))
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    from .fixed_experiments import run_fixed_examples

    keys = tuple(PROBLEMS) if args.problem == "all" else (args.problem,)
    metrics = run_fixed_examples(keys, args.profile, args.output, args.device, args.seed)
    if any(not item.no_worse_global or not item.no_worse_terminal for item in metrics):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
