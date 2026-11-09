"""Draw the PR-curve figure from saved results."""

import argparse
import json

import pandas as pd

from shoppers.config import load_config
from shoppers.plotting import plot_pr_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = json.loads((cfg.results_dir / "metrics.json").read_text())
    oof = pd.read_csv(cfg.results_dir / "oof_predictions.csv")
    plot_pr_curves(oof, metrics, cfg.figure_path)
    print(f"wrote {cfg.figure_path.name}")


if __name__ == "__main__":
    main()
