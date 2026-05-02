import json
import os
import pandas as pd


def compute_baseline(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    baseline = {
        "numeric": {
            "mean": numeric_df.mean().to_dict(),
            "std": numeric_df.std().to_dict(),
            "min": numeric_df.min().to_dict(),
            "max": numeric_df.max().to_dict(),
        },
        "categorical": {},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    for col in categorical_cols:
        baseline["categorical"][col] = df[col].value_counts(normalize=True).to_dict()

    return baseline


def save_baseline(baseline, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(baseline, f, indent=4)
    print(f"Baseline saved at: {path}")


def load_baseline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
