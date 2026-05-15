"""CLI entrypoint for training, evaluation, and explanations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit risk modeling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fit model and write models/*.pkl")
    p_train.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to processed parquet (default: data/processed/train.parquet)",
    )
    p_train.add_argument("--target", default="target", help="Binary target column name")

    p_eval = sub.add_parser("evaluate", help="Metrics on holdout parquet")
    p_eval.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to processed parquet (default: data/processed/test.parquet)",
    )
    p_eval.add_argument("--target", default="target")

    sub.add_parser("explain", help="Print top feature importances")

    args = parser.parse_args()

    if args.command == "train":
        from train_model import train

        train(processed_path=args.data, target_column=args.target)
    elif args.command == "evaluate":
        from evaluate_model import evaluate

        evaluate(processed_path=args.data, target_column=args.target)
    elif args.command == "explain":
        from explain_model import feature_importance

        for name, score in list(feature_importance().items())[:25]:
            print(f"{name}: {score:.6f}")


if __name__ == "__main__":
    main()
