# Data Drift Monitor v2.0

A modular, automated data drift monitoring system for ML models.
Detects feature drift, prediction drift, fires alerts, and shows a live dashboard — all automatically.

---

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

---

## File Structure

```
DATA_DRIFT_MONITOR/
├── driftmonitor/
│   ├── __init__.py                  ← exposes DriftMonitor
│   ├── driftmonitor.py              ← main class (start here)
│   ├── baseline/
│   │   └── baseline.py              ← compute / save / load baseline
│   ├── drift/
│   │   ├── psi.py                   ← PSI for numeric AND categorical (fixed)
│   │   ├── drift_statistical.py     ← KS test, categorical drift
│   │   └── global_drift.py          ← weighted global score (typo fixed)
│   ├── engine/
│   │   ├── common_engine.py         ← feature analyzer, severity, patterns
│   │   └── main_drift_runner.py     ← orchestrates per-column detection
│   ├── streaming/
│   │   └── streaming_engine.py      ← StreamBuffer + PredictionLogger (NEW)
│   ├── alerting/
│   │   └── alerting.py              ← AlertManager: console + Slack (NEW)
│   ├── automation/
│   │   └── automated_monitor.py     ← background thread scheduler (NEW)
│   ├── reporting/
│   │   └── report_generator.py      ← txt report + JSONL history (updated)
│   ├── visualization/
│   │   └── drift_visualizer.py      ← numeric, categorical, prediction plots
│   └── dashboard/
│       └── dashboard_app.py         ← Streamlit dashboard with 4 tabs (updated)
├── examples/
│   └── example_usage.py             ← batch / stream / automated examples
├── outputs/                         ← auto-created
│   ├── latest_results.json
│   ├── results_history.jsonl        ← full audit trail (NEW)
│   ├── prediction_log.csv           ← every prediction logged (NEW)
│   ├── baseline.json
│   ├── drift_report.txt
│   └── *.png                        ← drift plots
├── requirements.txt
└── setup.py
```

---

## Quick Start

### Mode 1 — Batch (simplest)

```python
from driftmonitor import DriftMonitor
import pandas as pd

train_df = pd.read_csv("examples/data/train.csv")
new_df   = pd.read_csv("examples/data/new_batch.csv")

monitor = DriftMonitor(
    weights={"amount": 3, "category": 2, "age": 1}
)
monitor.fit(train_df)

result = monitor.detect(new_df)
monitor.visualize(new_df)
monitor.generate_report(
    result["feature_results"],
    result["global_score"],
    result["health"],
    result["action"]
)
monitor.launch_dashboard()
```

---

### Mode 2 — Stream (record by record)

```python
monitor = DriftMonitor(batch_size=50)
monitor.fit(train_df)

for record in live_stream:
    features   = {k: v for k, v in record.items()}
    prediction = model.predict([features])[0]

    # Logs to CSV + auto-triggers drift check every 50 records
    monitor.log_prediction(features, prediction)
```

---

### Mode 3 — Fully Automated (recommended for production)

```python
monitor = DriftMonitor(
    weights={"amount": 3, "category": 2, "age": 1},
    slack_webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ"
)
monitor.fit(train_df, train_predictions=model.predict(train_df))

# Start background monitoring thread
monitor.start_auto_monitoring(
    check_every_n_records=100,   # OR: interval_seconds=3600
    generate_report=True,
    visualize=True,
)

monitor.launch_dashboard()  # open http://localhost:8501

# Your model runs normally — just log each prediction
for record in production_stream:
    prediction = model.predict([record])[0]
    monitor.log_prediction(record, prediction)

monitor.stop_auto_monitoring()
```

---

## Slack Alerts

1. Go to https://api.slack.com/messaging/webhooks
2. Create an incoming webhook for your channel
3. Pass the URL to `DriftMonitor(slack_webhook_url="https://hooks.slack.com/...")`

Alerts fire automatically on `CRITICAL` or `WARNING` health states.

---

## Dashboard

The Streamlit dashboard has 4 tabs:

| Tab | What it shows |
|-----|---------------|
| Latest Check | Current drift scores, feature table with severity colouring |
| History | Line chart of global drift score over time |
| Prediction Log | Every logged prediction, class distribution chart |
| Drift Plots | Feature drift plots + prediction drift plot |

Auto-refreshes every 30 seconds (install `streamlit-autorefresh`).

---

## Health Thresholds

| Global Score | Health | Action |
|---|---|---|
| < 0.3 | 🟢 HEALTHY | Continue Monitoring |
| 0.3 – 0.6 | 🟡 WARNING | Investigate the Data |
| > 0.6 | 🔴 CRITICAL | Retrain the Model |

---

## Key Fixes in v2.0

| File | What was fixed |
|---|---|
| `psi.py` | Separate `calculate_categorical_psi()` — no longer re-histograms pre-binned proportions |
| `global_drift.py` | Typo `comput_` → `compute_` fixed; configurable weights |
| `drift_statistical.py` | Uses correct PSI function for categorical data |
| `main_drift_runner.py` | Hardcoded weights removed; configurable via `DriftMonitor` |
| `driftmonitor.py` | `fit() firts` typo fixed; weights, alerting, auto-monitoring added |
| `drift_visualizer.py` | Filename typo `drfift_` fixed; `plot_prediction_drift()` added |
| `dashboard_app.py` | 4 tabs, auto-refresh, history chart, prediction log view |
| `streaming_engine.py` | `PredictionLogger` added for persistent CSV logging |
