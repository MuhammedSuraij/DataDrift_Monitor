import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Agg = file renderer, no GUI window needed
import matplotlib.pyplot as plt


def plot_numeric_drift(train_col, new_col, feature_name, output_dir="outputs"):
    """Plot density curves for a numeric feature: baseline vs new data."""
    os.makedirs(output_dir, exist_ok=True)

    train_col = train_col.dropna()
    new_col   = new_col.dropna()

    if len(train_col) == 0 or len(new_col) == 0:
        print(f"  [SKIP] {feature_name}: empty column, skipping plot.")
        return None

    train_density = np.histogram(train_col, bins=40, density=True)
    new_density   = np.histogram(new_col,   bins=40, density=True)

    train_x = (train_density[1][1:] + train_density[1][:-1]) / 2
    new_x   = (new_density[1][1:]   + new_density[1][:-1])   / 2

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_x, train_density[0], label="Baseline (Training)", linewidth=2, color="#2E86AB")
    ax.plot(new_x,   new_density[0],   label="New Incoming Data",   linewidth=2, linestyle="--", color="#E84855")

    ax.set_title(f"Data Drift — '{feature_name}'", fontsize=13, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    file_path = os.path.join(output_dir, f"{feature_name}_numeric_drift.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=120)
    plt.close(fig)   # always close to free memory

    return file_path


def plot_categorical_drift(train_col, new_col, feature_name, output_dir="outputs"):
    """Plot category proportion comparison for a categorical feature."""
    os.makedirs(output_dir, exist_ok=True)

    train_dist = train_col.value_counts(normalize=True)
    new_dist   = new_col.value_counts(normalize=True)

    df = pd.DataFrame({"Baseline": train_dist, "New": new_dist}).fillna(0).sort_index()

    if df.empty:
        print(f"  [SKIP] {feature_name}: no data, skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df.index, df["Baseline"], marker="o", label="Baseline (Training)",
            linewidth=2, color="#2E86AB")
    ax.plot(df.index, df["New"],      marker="o", linestyle="--", label="New Incoming Data",
            linewidth=2, color="#E84855")

    ax.set_title(f"Category Drift — '{feature_name}'", fontsize=13, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    file_path = os.path.join(output_dir, f"{feature_name}_categorical_drift.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=120)
    plt.close(fig)

    return file_path


def plot_prediction_drift(train_predictions, new_predictions, output_dir="outputs"):
    """Plot prediction distribution: baseline vs new."""
    os.makedirs(output_dir, exist_ok=True)

    train_pred = np.array(train_predictions)
    new_pred   = np.array(new_predictions)

    fig, ax = plt.subplots(figsize=(9, 4))

    if len(np.unique(train_pred)) <= 10:
        all_labels   = sorted(set(train_pred) | set(new_pred))
        train_counts = [np.mean(train_pred == lb) for lb in all_labels]
        new_counts   = [np.mean(new_pred   == lb) for lb in all_labels]
        x = np.arange(len(all_labels))
        ax.bar(x - 0.2, train_counts, 0.4, label="Baseline", color="#2E86AB", alpha=0.8)
        ax.bar(x + 0.2, new_counts,   0.4, label="New Data",  color="#E84855", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels)
        ax.set_ylabel("Proportion")
    else:
        t_hist = np.histogram(train_pred, bins=40, density=True)
        n_hist = np.histogram(new_pred,   bins=40, density=True)
        t_x = (t_hist[1][1:] + t_hist[1][:-1]) / 2
        n_x = (n_hist[1][1:] + n_hist[1][:-1]) / 2
        ax.plot(t_x, t_hist[0], label="Baseline", linewidth=2, color="#2E86AB")
        ax.plot(n_x, n_hist[0], label="New Data",  linewidth=2, linestyle="--", color="#E84855")
        ax.set_ylabel("Density")

    ax.set_title("Prediction Drift", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediction")
    ax.legend()
    ax.grid(alpha=0.3)

    file_path = os.path.join(output_dir, "prediction_drift.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=120)
    plt.close(fig)

    return file_path