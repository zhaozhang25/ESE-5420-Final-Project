"""Cross-validated scoring. All functions here work on dev data only."""

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

from .models import make_logreg

SCORING = "average_precision"


def summarize(scores):
    """Mean, SD and SE = SD / sqrt(K) of per-fold scores."""
    scores = np.asarray(scores, dtype=float)
    sd = scores.std(ddof=1)
    return {"mean": float(scores.mean()), "sd": float(sd),
            "se": float(sd / np.sqrt(len(scores))),
            "folds": [float(s) for s in scores]}


def floor_score(X, y, folds):
    """AUPRC of the always-no-purchase rule, which equals each fold's purchase rate."""
    scores = cross_val_score(DummyClassifier(strategy="prior"), X, y,
                             cv=folds, scoring=SCORING)
    return summarize(scores)


def tune_logreg(X, y, folds, use_pagevalues, c_grid, max_iter):
    """Choose C by the highest mean validation AUPRC.

    Returns the summary for the chosen C and the mean score at each grid value.
    Raises if the chosen C is on the edge of the grid or the solver fails to converge.
    """
    search = GridSearchCV(make_logreg(use_pagevalues, max_iter), {"clf__C": list(c_grid)},
                          cv=folds, scoring=SCORING, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        search.fit(X, y)
    best_c = float(search.best_params_["clf__C"])
    if best_c in (float(c_grid[0]), float(c_grid[-1])):
        raise ValueError(f"best C = {best_c} is on the grid edge; widen the grid")
    i = search.best_index_
    fold_scores = [search.cv_results_[f"split{k}_test_score"][i] for k in range(len(folds))]
    return {"best_C": best_c, **summarize(fold_scores)}, search.cv_results_["mean_test_score"]


def shuffled_label_check(X, y, c, n_folds, seed, max_iter, n_shuffles=20):
    """With shuffled labels a sound pipeline should score about what chance scores.

    Reports the CV AUPRC over repeated shuffles, since one shuffle is noisy, and
    the AUPRC of random scores on the same shuffled folds as the chance reference.
    `se` is the standard error of the mean over shuffles.
    """
    rng = np.random.default_rng(seed)
    model = make_logreg(True, max_iter).set_params(clf__C=c)
    model_scores, chance_scores = [], []
    for _ in range(n_shuffles):
        shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        model_scores.append(cross_val_score(model, X, shuffled, cv=cv, scoring=SCORING).mean())
        chance_scores.append(np.mean([
            average_precision_score(shuffled.iloc[va], rng.random(len(va)))
            for _, va in cv.split(X, shuffled)]))
    return {"n_shuffles": n_shuffles, **summarize(model_scores),
            "chance_mean": float(np.mean(chance_scores))}


def out_of_fold_scores(X, y, folds, use_pagevalues, c, max_iter):
    """Predict each dev row with a model that did not train on it."""
    model = make_logreg(use_pagevalues, max_iter).set_params(clf__C=c)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
