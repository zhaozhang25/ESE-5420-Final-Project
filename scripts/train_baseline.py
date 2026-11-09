"""Tune logistic regression on the dev folds and save its results. Scores no test data."""

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn

from shoppers.config import load_config
from shoppers.data import LEAKAGE_SUSPECT, load_dataset, load_raw
from shoppers.evaluate import (floor_score, out_of_fold_scores, shuffled_label_check,
                               summarize, tune_logreg)
from shoppers.splits import load_dev, read_splits

VARIANTS = {"with_pagevalues": True, "without_pagevalues": False}


def data_profile(cfg, X_dev, y_dev):
    """Dataset counts, plus the PageValues statistics computed on dev rows only."""
    raw = load_raw(cfg.data_path)
    duplicates = raw[raw.duplicated(keep="first")]
    pv = X_dev[LEAKAGE_SUSPECT]
    return {
        "n_raw": len(raw),
        "n_duplicates": len(duplicates),
        "duplicates_that_are_purchases": int(duplicates.Revenue.sum()),
        "duplicates_that_are_bounces": int(
            ((duplicates.BounceRates == 0.2) & (duplicates.ExitRates == 0.2)).sum()),
        "dev_p_pagevalues_positive_given_purchase": float((pv[y_dev == 1] > 0).mean()),
        "dev_p_pagevalues_positive_given_no_purchase": float((pv[y_dev == 0] > 0).mean()),
        "dev_p_purchase_given_pagevalues_positive": float(y_dev[pv > 0].mean()),
        "dev_p_purchase_given_pagevalues_zero": float(y_dev[pv == 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    X, y = load_dataset(cfg.data_path, cfg.drop_duplicates)
    table = read_splits(cfg.splits_path)
    X_dev, y_dev, folds = load_dev(X, y, table)

    results, curves = {}, {"C": cfg.c_grid}
    oof = pd.DataFrame({"row": X_dev.index, "y": y_dev.to_numpy()})
    for name, use_pv in VARIANTS.items():
        results[name], curves[name] = tune_logreg(
            X_dev, y_dev, folds, use_pv, cfg.c_grid, cfg.max_iter)
        oof[f"score_{name}"] = out_of_fold_scores(
            X_dev, y_dev, folds, use_pv, results[name]["best_C"], cfg.max_iter)

    metrics = {
        "config": cfg.name,
        "seed": cfg.seed,
        "n": len(X),
        "n_inputs": X.shape[1],
        "n_dev": len(X_dev),
        "n_test": len(X) - len(X_dev),
        "purchase_rate": float(y.mean()),
        "floor": floor_score(X_dev, y_dev, folds),
        **results,
        "pagevalues_effect": summarize(np.subtract(
            results["with_pagevalues"]["folds"], results["without_pagevalues"]["folds"])),
        "shuffled_label_check": shuffled_label_check(
            X_dev, y_dev, results["with_pagevalues"]["best_C"],
            cfg.n_folds, cfg.seed, cfg.max_iter),
        "data_profile": data_profile(cfg, X_dev, y_dev),
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(curves).to_csv(cfg.results_dir / "cv_curve.csv", index=False)
    oof.to_csv(cfg.results_dir / "oof_predictions.csv", index=False)

    for name in VARIANTS:
        r = results[name]
        print(f"{cfg.name}/{name}: C = {r['best_C']:.4g}, "
              f"CV AUPRC = {r['mean']:.4f} +/- {r['se']:.4f} (SE)")
    print(f"{cfg.name}/floor: {metrics['floor']['mean']:.4f}")


if __name__ == "__main__":
    main()
