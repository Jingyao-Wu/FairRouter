"""Command line entry point. Import numerical libraries only inside Slurm."""

import argparse
from pathlib import Path

from .artifacts import DATASETS, SEEDS, SHOTS, require_slurm


def main():
    parser = argparse.ArgumentParser(
        prog="fairrouter", description="Run FairRouter for few-shot node classification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Train or predict for one dataset and split")
    run.add_argument("--experts", type=Path, help="Optional prepared GCN head outputs")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument(
        "--banks", type=Path, help="Optional routing features computed from expert representations"
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--shot", type=int, choices=SHOTS)
    run.add_argument("--seed", type=int, choices=SEEDS)
    run.add_argument("--dataset", choices=DATASETS)
    run.add_argument("--index", type=int, help="Array index 0..44, shot/seed/dataset order")
    run.add_argument(
        "--mode", choices=("replay", "refit-joint", "refit-all"), default="refit-joint"
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate all datasets, label budgets and seeds"
    )
    evaluate.add_argument("--experts", type=Path)
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path)
    verify = subparsers.add_parser("verify", help="Check the prepared data and model files")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    require_slurm()
    if args.command == "run":
        from .pipeline import run_cell

        if args.index is not None:
            cells = [(s, r, d) for s in SHOTS for r in SEEDS for d in DATASETS]
            if not 0 <= args.index < len(cells):
                parser.error("--index must be 0..44")
            args.shot, args.seed, args.dataset = cells[args.index]
        if args.shot is None or args.seed is None or args.dataset is None:
            parser.error("Supply --index or --shot/--seed/--dataset")
        run_cell(
            args.bundle,
            args.output,
            args.shot,
            args.seed,
            args.dataset,
            args.mode,
            args.banks,
            args.experts,
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_run

        evaluate_run(args.bundle, args.run, args.output, args.reference, args.experts)
    else:
        from .artifacts import Bundle

        bundle = Bundle(args.bundle)
        for digest in bundle.manifest["objects"]:
            bundle.object(digest)
        for cell in bundle.manifest["cells"]:
            bundle.split(cell)
        print("Verified", len(bundle.manifest["objects"]), "objects and 45 splits")


if __name__ == "__main__":
    main()
