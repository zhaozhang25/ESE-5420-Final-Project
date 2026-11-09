# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ESE 5420 final project: tuned MLP vs. tuned logistic regression on test AUPRC (UCI Online Shoppers). Every result is reported with and without `PageValues`.

## Commands

```bash
python scripts/train_baseline.py --config configs/baseline.yaml   # also keep_duplicates.yaml
python scripts/make_figures.py   --config configs/baseline.yaml
pytest tests/test_pipeline.py::test_name   # single test (tests/ is local only, gitignored)
```

## Rules

- New models go through `shoppers.models.make_pipeline(classifier, use_pagevalues)` and the folds from `shoppers.splits.load_dev`.
- Never score the test set while tuning. It is scored once, at the end.
- Don't overwrite `data/splits/` or edit `data/raw/`.
- `tests/test_artifacts.py` checks that the numbers in `README.md` and `reports/checkpoint/baseline.txt` match `results/baseline/metrics.json`. If results change, update those files.
