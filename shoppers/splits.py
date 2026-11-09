"""The shared train/test split and CV folds.

Every model in the project reads the same split file. This module deliberately
has no function that returns test rows: `load_dev` is the only way to get data
from a split, so tuning code cannot reach the test set by accident.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def compute_splits(y, test_size, n_folds, seed):
    """Return a table with columns row, role ("dev"/"test") and fold (-1 for test)."""
    dev_rows, test_rows = train_test_split(
        y.index.to_numpy(), test_size=test_size, stratify=y, random_state=seed)
    y_dev = y.loc[dev_rows]
    fold = np.full(len(dev_rows), -1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y_dev)), y_dev)):
        fold[va] = k
    table = pd.concat([
        pd.DataFrame({"row": dev_rows, "role": "dev", "fold": fold}),
        pd.DataFrame({"row": test_rows, "role": "test", "fold": -1}),
    ])
    return table.sort_values("row").reset_index(drop=True)


def write_splits(table, path, force=False):
    """Save a split, refusing to replace a different existing split unless forced."""
    if path.exists() and not force and not pd.read_csv(path).equals(table):
        raise FileExistsError(
            f"{path} holds a different split. Results already depend on it; "
            "pass force=True only if the whole team agrees to rerun everything.")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def read_splits(path):
    return pd.read_csv(path)


def load_dev(X, y, table):
    """Return dev inputs, dev labels and folds as (train, validation) position arrays."""
    if set(table["row"]) != set(X.index):
        raise ValueError("split file does not match the dataset rows; check drop_duplicates")
    dev = table[table["role"] == "dev"].sort_values("row")
    fold = dev["fold"].to_numpy()
    folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k))
             for k in sorted(set(fold))]
    return X.loc[dev["row"]], y.loc[dev["row"]], folds
