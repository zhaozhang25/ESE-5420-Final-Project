"""Load the UCI Online Shoppers Purchasing Intention dataset."""

import pandas as pd

TARGET = "Revenue"
LEAKAGE_SUSPECT = "PageValues"
NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]


def load_raw(path):
    """Read the CSV and check that it has the expected columns."""
    raw = pd.read_csv(path)
    expected = set(NUMERIC) | set(CATEGORICAL) | {TARGET}
    if set(raw.columns) != expected:
        raise ValueError(f"unexpected columns: {sorted(set(raw.columns) ^ expected)}")
    return raw


def load_dataset(path, drop_duplicates):
    """Return inputs X and label y. The index is the row's position in the CSV (from 0)."""
    raw = load_raw(path)
    if drop_duplicates:
        raw = raw[~raw.duplicated(keep="first")]
    return raw.drop(columns=TARGET), raw[TARGET].astype(int)
