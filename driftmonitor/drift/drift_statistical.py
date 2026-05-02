from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import pandas as pd
import numpy as np
from driftmonitor.drift.psi import calculate_psi, calculate_categorical_psi


def is_numeric(series):
    return series.dtype in ["int64", "float64"]


def ks_drift(col_train, col_new):
   
    stat, p = ks_2samp(col_train.dropna(), col_new.dropna())
    return p < 0.05, float(p)


def wasserstein_drift(col_train, col_new):
    
    train = col_train.dropna().values.astype(float)
    new   = col_new.dropna().values.astype(float)

    if len(train) == 0 or len(new) == 0:
        return False, 0.0

    wd    = wasserstein_distance(train, new)
    std   = train.std()
    score = float(wd / std) if std > 0 else 0.0
    return score > 0.20, score


def categorical_drift(col_train, col_new):
    
    train_dist = col_train.value_counts(normalize=True)
    new_dist   = col_new.value_counts(normalize=True)

    all_categories = sorted(set(train_dist.index).union(set(new_dist.index)))
    expected = [train_dist.get(cat, 0.0) for cat in all_categories]
    actual   = [new_dist.get(cat, 0.0)   for cat in all_categories]

    psi = calculate_categorical_psi(expected, actual)

    train_counts = col_train.value_counts()
    new_counts   = col_new.value_counts()
    combined = pd.concat([train_counts, new_counts], axis=1).fillna(0)

    try:
        chi2, p, _, _ = chi2_contingency(combined)
        p = float(p)
    except Exception:
        p = 1.0

    drift = (psi > 0.25) or (p < 0.05)
    return drift, float(psi), p