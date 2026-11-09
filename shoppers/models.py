"""Model pipelines. Preprocessing sits inside each pipeline, so CV refits it per fold."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, LEAKAGE_SUSPECT, NUMERIC


def make_preprocessor(use_pagevalues):
    """Scale numeric inputs and one-hot encode categoricals. Unlisted columns are dropped.

    Some categories have a single row and can be absent from a training fold,
    so the encoder must ignore unseen levels.
    """
    numeric = [c for c in NUMERIC if use_pagevalues or c != LEAKAGE_SUSPECT]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def make_pipeline(classifier, use_pagevalues):
    """Attach any scikit-learn classifier to the shared preprocessing."""
    return Pipeline([("pre", make_preprocessor(use_pagevalues)), ("clf", classifier)])


def make_logreg(use_pagevalues, max_iter=5000):
    """L2-penalized logistic regression (the scikit-learn default penalty)."""
    return make_pipeline(LogisticRegression(max_iter=max_iter), use_pagevalues)
