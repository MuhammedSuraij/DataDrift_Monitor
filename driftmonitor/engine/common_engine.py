import pandas as pd
from driftmonitor.drift.drift_statistical import (
    is_numeric, ks_drift, wasserstein_drift, categorical_drift
)


# ── Severity ──────────────────────────────────────────────────────

def numeric_severity(ks_p, wasserstein_score):
    
    if ks_p < 0.01 or wasserstein_score > 1.0:
        return "HIGH"
    elif ks_p < 0.05 or wasserstein_score > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def psi_severity(psi):
    if psi >= 0.25:
        return "HIGH"
    elif psi >= 0.1:
        return "MEDIUM"
    else:
        return "LOW"



def drift_pattern(mean_change, std_change):
    if abs(mean_change) > 50 and abs(std_change) < 10:
        return "DATA SHIFT"
    elif abs(std_change) > 50:
        return "VARIANCE EXPLOSION"
    elif abs(mean_change) < 10 and abs(std_change) < 10:
        return "MINOR FLUCTUATION"
    else:
        return "COMPLEX DRIFT"


def numeric_drift_reason(train_col, new_col):
    if not (pd.api.types.is_numeric_dtype(train_col) and
            pd.api.types.is_numeric_dtype(new_col)):
        return None

    train_col = train_col.dropna()
    new_col   = new_col.dropna()

    train_mean = train_col.mean()
    train_std  = train_col.std()

    mean_change = ((new_col.mean() - train_mean) / train_mean * 100
                   if train_mean != 0 else 0.0)
    std_change  = ((new_col.std()  - train_std)  / train_std  * 100
                   if train_std  != 0 else 0.0)

    return {
        "mean_change_%": round(mean_change, 2),
        "std_change_%":  round(std_change,  2),
        "old_mean":      round(float(train_mean), 2),
        "new_mean":      round(float(new_col.mean()), 2),
        "old_range":     (float(train_col.min()), float(train_col.max())),
        "new_range":     (float(new_col.min()),   float(new_col.max())),
        "pattern":       drift_pattern(mean_change, std_change),
    }


def detect_new_categories(train_col, new_col):
    return list(set(new_col.unique()) - set(train_col.unique()))



def format_numeric_reason(reason):
    """Human-readable explanation of a numeric drift. Only shown on drift."""
    lines = [
        f"Mean: {reason['old_mean']} (training) → {reason['new_mean']} (new)  "
        f"({'+' if reason['mean_change_%'] >= 0 else ''}{reason['mean_change_%']}%)",
        f"Std changed by {reason['std_change_%']}%",
        f"Old range: {reason['old_range']}   New range: {reason['new_range']}",
        f"Pattern: {reason['pattern']}",
    ]
    return lines


def format_categorical_reason(new_categories, train_col, new_col):
    """Human-readable explanation of a categorical drift. Only shown on drift."""
    lines = []
    if new_categories:
        lines.append(f"New unseen categories: {', '.join(map(str, new_categories))}")

    train_dist = train_col.value_counts(normalize=True)
    new_dist   = new_col.value_counts(normalize=True)
    all_cats   = set(train_dist.index) | set(new_dist.index)
    changes = []
    for cat in all_cats:
        old_pct = train_dist.get(cat, 0) * 100
        new_pct = new_dist.get(cat, 0) * 100
        changes.append((cat, old_pct, new_pct, abs(new_pct - old_pct)))

    changes.sort(key=lambda x: x[3], reverse=True)
    for cat, old_pct, new_pct, delta in changes[:3]:
        if delta > 1.0:
            direction = "up" if new_pct > old_pct else "down"
            lines.append(
                f"'{cat}': {old_pct:.1f}% → {new_pct:.1f}%  ({direction} {delta:.1f}%)"
            )
    return lines



def analyze_feature_drift(train_df, new_df, col):
    result = {"feature": col}

    if is_numeric(train_df[col]):
        ks_drifted,  ks_p    = ks_drift(train_df[col], new_df[col])
        wd_drifted,  wd_score = wasserstein_drift(train_df[col], new_df[col])

        drifted  = ks_drifted or wd_drifted
        severity = numeric_severity(ks_p, wd_score)

        ks_score  = round(1 - ks_p, 4)
        combined  = round((ks_score + min(wd_score, 1.0)) / 2, 4)

        reason = numeric_drift_reason(train_df[col], new_df[col])

        result.update({
            "type":         "numeric",
            "method":       "KS + Wasserstein",
            "drift_detected": drifted,
            "severity":     severity,
            "score":        combined,
            "explanation":  format_numeric_reason(reason) if (drifted and reason) else [],
        })

    else:
        drifted, psi, p_chi = categorical_drift(train_df[col], new_df[col])
        severity    = psi_severity(psi)
        new_cats    = detect_new_categories(train_df[col], new_df[col])

        result.update({
            "type":         "categorical",
            "method":       "PSI + Chi-Square",
            "drift_detected": drifted,
            "severity":     severity,
            "score":        round(psi, 4),
            "explanation":  format_categorical_reason(new_cats, train_df[col], new_df[col])
                            if drifted else [],
        })

    return result


# ── Health & Action ───────────────────────────────────────────────

def system_health(global_score):
    if global_score > 0.6:
        return "CRITICAL"
    elif global_score > 0.3:
        return "WARNING"
    else:
        return "HEALTHY"


def recommended_action(health):
    if health == "CRITICAL":
        return "Retrain the Model"
    elif health == "WARNING":
        return "Investigate the Data"
    else:
        return "Continue Monitoring"