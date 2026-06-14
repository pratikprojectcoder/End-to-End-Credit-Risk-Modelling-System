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

    sub.add_parser("seed", help="Create demo model artifacts for API/dashboard")

    p_serve = sub.add_parser("serve", help="Start FastAPI server with real-time dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")

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
    elif args.command == "seed":
        from seed_models import seed_demo_models

        seed_demo_models(force=True)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
