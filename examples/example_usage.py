"""
╔══════════════════════════════════════════════════════════════════╗
║      DATA DRIFT MONITOR — US ADULT INCOME DATASET               ║
║                                                                  ║
║  Baseline : adult_1994.csv  (32,561 people surveyed in 1994)     ║
║  New Batch: adult_2000.csv  (year-2000 data)                     ║
║                                                                  ║
║  Target   : Does this person earn >50K/year? (Yes=1 / No=0)     ║
║  Model    : Gradient Boosting Classifier (~85% accuracy)         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import time
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from driftmonitor.driftmonitor import DriftMonitor

FEATURE_COLS = [
    "age", "education_num", "hours_per_week",
    "capital_gain", "capital_loss",
    "workclass", "marital_status", "occupation", "relationship", "sex",
]
TARGET_COL  = "income"
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "sex"]
NUMERIC     = [c for c in FEATURE_COLS if c not in CATEGORICAL]

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH     = os.path.join(BASE_DIR, "data", "adult_1994.csv")
NEW_BATCH_PATH = os.path.join(BASE_DIR, "data", "adult_2000.csv")
OUTPUTS_DIR    = os.path.join(ROOT, "outputs")


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1: Loading Adult Income data")
print("="*60)

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    df = df[df["occupation"] != "?"].dropna(subset=FEATURE_COLS)
    return df.reset_index(drop=True)

train_raw = load_and_clean(TRAIN_PATH)
new_raw   = load_and_clean(NEW_BATCH_PATH)

print(f"  1994 Baseline : {len(train_raw):,} people")
print(f"  2000 New Batch: {len(new_raw):,} people")
print(f"\n  Income >50K rate:")
print(f"    1994: {(train_raw[TARGET_COL]=='>50K').mean()*100:.1f}%")
print(f"    2000: {(new_raw[TARGET_COL]=='>50K').mean()*100:.1f}%")
print(f"\n  Capital gain mean:")
print(f"    1994: ${train_raw['capital_gain'].mean():,.0f}")
print(f"    2000: ${new_raw['capital_gain'].mean():,.0f}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — CREATE LABELS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2: Creating binary labels")
print("="*60)

train_raw["label"] = (train_raw[TARGET_COL] == ">50K").astype(int)
new_raw["label"]   = (new_raw[TARGET_COL]   == ">50K").astype(int)

count_0 = (train_raw["label"] == 0).sum()
count_1 = (train_raw["label"] == 1).sum()
print(f"  Label 0 (<=50K): {count_0:,}  ({count_0/len(train_raw)*100:.0f}%)")
print(f"  Label 1  (>50K): {count_1:,}  ({count_1/len(train_raw)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — ENCODE CATEGORICALS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3: Encoding categorical columns")
print("="*60)

encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder().fit(train_raw[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {list(le.classes_)[:4]}{'...' if len(le.classes_)>4 else ''}")

def prepare_features(df):
    d = df[FEATURE_COLS].copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        d[col] = d[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        d[col] = le.transform(d[col])
    return d.astype(float)

X_all = prepare_features(train_raw)
y_all = train_raw["label"]


# ══════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4: Training Gradient Boosting model")
print("="*60)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)



model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
model.fit(X_tr, y_tr)

accuracy = accuracy_score(y_te, model.predict(X_te))
print(f"  Accuracy : {accuracy*100:.1f}%")
print()
print(classification_report(y_te, model.predict(X_te),
      target_names=["<=50K (0)", ">50K (1)"], zero_division=0))

imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("  Top 5 most important features:")
for f, v in imp.head(5).items():
    print(f"    {f:20s}: {v:.3f}")

train_predictions = model.predict(X_all).tolist()
print(f"\n  Training prediction distribution:")
print(f"    Predicted <=50K: {train_predictions.count(0):,} ({train_predictions.count(0)/len(train_predictions)*100:.0f}%)")
print(f"    Predicted  >50K: {train_predictions.count(1):,} ({train_predictions.count(1)/len(train_predictions)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE ON NEW (DRIFTED) DATA
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5: Evaluating on NEW (year 2000) data")
print("="*60)

# Load + clean (same as training)
new_test = load_and_clean(NEW_BATCH_PATH)

# Create labels
new_test["label"] = (new_test[TARGET_COL] == ">50K").astype(int)

# Prepare features using SAME encoders
X_new = prepare_features(new_test)
y_new = new_test["label"]

# Predict
y_pred_new = model.predict(X_new)

# Accuracy
new_accuracy = accuracy_score(y_new, y_pred_new)
print(f"  2000 data Accuracy : {new_accuracy*100:.2f}%")

# Detailed report
print("\n  Classification Report (New Data):\n")
print(classification_report(
    y_new, y_pred_new,
    target_names=["<=50K (0)", ">50K (1)"],
    zero_division=0
))

# Prediction distribution
pred_list = y_pred_new.tolist()
print("\n  Prediction Distribution (New Data):")
print(f"    Predicted <=50K: {pred_list.count(0):,} ({pred_list.count(0)/len(pred_list)*100:.0f}%)")
print(f"    Predicted  >50K: {pred_list.count(1):,} ({pred_list.count(1)/len(pred_list)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# STEP 5 — SET UP DRIFT MONITOR
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5: Setting up Drift Monitor")
print("="*60)

monitor = DriftMonitor(
    batch_size=1000,
    weights={
        "capital_gain":   4,
        "age":            2,
        "education_num":  2,
        "hours_per_week": 1,
        "capital_loss":   1,
        "workclass":      2,
        "marital_status": 1,
        "occupation":     2,
        "relationship":   1,
        "sex":            1,
    },
    slack_webhook_url='slack_url_here',
    alert_on=("CRITICAL", "WARNING"),
)

monitor.reset_outputs()

train_features_raw = train_raw[FEATURE_COLS].copy()
monitor.fit(train_features_raw, train_predictions=train_predictions)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
monitor.save_baseline(os.path.join(OUTPUTS_DIR, "baseline.json"))
print("  Monitor fitted on 1994 baseline data.")
print("  Baseline saved -> outputs/baseline.json")


# ══════════════════════════════════════════════════════════════════
# STEP 6 — START AUTOMATED BACKGROUND MONITORING
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6: Starting automated background monitoring")
print("="*60)

monitor.start_auto_monitoring(
    check_every_n_records=1000,
    generate_report=True,
    visualize=True,
    feature_columns=FEATURE_COLS,
)
print("  Background monitor started. Drift check fires every 1000 predictions.")


# ══════════════════════════════════════════════════════════════════

def _force_drift_check(monitor, feature_cols):
    """Run detect() on whatever is currently in the prediction log."""
    try:
        log_df = monitor.logger.load()
        if log_df.empty:
            print("  No predictions logged yet.")
            return
        missing = [c for c in feature_cols if c not in log_df.columns]
        if missing:
            print(f"  Cannot run check — missing columns: {missing}")
            return
        new_df = log_df[feature_cols].copy()
        result = monitor.detect(new_df)
        health = result["health"]
        score  = round(result["global_score"], 4)
        icons  = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}
        print(f"\n  {icons.get(health,'?')} Health: {health}  |  Global Score: {score}")
        drifted = [r["feature"] for r in result["feature_results"] if r["drift_detected"]]
        if drifted:
            print(f"  Drifted features: {', '.join(drifted)}")
        else:
            print("  No significant drift detected in this batch.")
        print("  Dashboard updated. Check http://localhost:8501")
    except Exception as e:
        print(f"  (Final drift check failed: {e})")




# STEP 7 — LAUNCH DASHBOARD (BEFORE mode choice)
# The dashboard opens NOW and auto-refreshes every 30 seconds.
# It updates automatically whenever a drift check fires.
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 7: Launching dashboard")
print("="*60)

# FIX 3: Dashboard launches HERE — before any mode is chosen.
# This means it is already running when you start entering data.
# We wait 3 seconds so Streamlit has time to start before the
# terminal prompts appear.
monitor.launch_dashboard()
print("  Dashboard is starting — open http://localhost:8501")
print("  It auto-refreshes every 30 seconds as predictions come in.")
time.sleep(3)


# ══════════════════════════════════════════════════════════════════
# STEP 8 — CHOOSE MODE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CHOOSE MODE")
print("="*60)
print()
print("  [1] BATCH       — run automatically on 2000 data")
print("  [2] INTERACTIVE — type a person's details manually")
print()

while True:
    mode = input("  Enter 1 or 2: ").strip()
    if mode in ("1", "2"):
        break
    print("  Please type 1 or 2.")


# ══════════════════════════════════════════════════════════════════
# MODE 1 — BATCH
# ══════════════════════════════════════════════════════════════════
if mode == "1":
    print("\n" + "="*60)
    print("  BATCH MODE: Running 2000 data through model")
    print("="*60)

    X_new     = prepare_features(new_raw)
    all_preds = model.predict(X_new)
    pred_0 = pred_1 = 0

    for i, (_, raw_row) in enumerate(new_raw.iterrows()):
        prediction = int(all_preds[i])
        if prediction == 0: pred_0 += 1
        else:               pred_1 += 1

        monitor.log_prediction(
            features=raw_row[FEATURE_COLS].to_dict(),
            prediction=prediction,
        )

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1:5d}/{len(new_raw):,}]  <=50K: {pred_0}  >50K: {pred_1}")
        time.sleep(0.005)

    total = pred_0 + pred_1
    print(f"\n  Done. {total:,} people scored.")
    print(f"  Predicted <=50K: {pred_0:,} ({pred_0/total*100:.0f}%)")
    print(f"  Predicted  >50K: {pred_1:,} ({pred_1/total*100:.0f}%)")
    print(f"\n  Compare to 1994 training:")
    print(f"    Training <=50K: {train_predictions.count(0)/len(train_predictions)*100:.0f}%")
    print(f"    Training  >50K: {train_predictions.count(1)/len(train_predictions)*100:.0f}%")


# ══════════════════════════════════════════════════════════════════
# MODE 2 — INTERACTIVE
# ══════════════════════════════════════════════════════════════════
else:
    print("\n" + "="*60)
    print("  INTERACTIVE MODE — Describe a person")
    print("="*60)
    print()
    print("  HOW DASHBOARD UPDATES IN INTERACTIVE MODE:")
    print("  +----------------------------------------------------------+")
    print("  | 1. You enter a person's details -> prediction shown      |")
    print("  | 2. That prediction is saved to prediction_log.csv        |")
    print("  | 3. After every 1000 entries -> drift check fires           |")
    print("  | 4. Dashboard auto-refreshes every 30 seconds             |")
    print("  | 5. Check http://localhost:8501 to see updated results    |")
    print("  +----------------------------------------------------------+")
    print()
    print("  Categorical fields accept ANY value.")
    print("  Typing a new value (e.g. 'AI-engineer') triggers drift.")
    print()
    print("  Type 'done' at any prompt to stop early.")
    print()

    FIELDS = [
        ("Age [17-90]",               "age",            "int", (17, 90)),
        ("Education years [1-16]",    "education_num",  "int", (1, 16)),
        ("Hours per week [1-99]",     "hours_per_week", "int", (1, 99)),
        ("Capital gain [0-99999]",    "capital_gain",   "int", (0, 99999)),
        ("Capital loss [0-4356]",     "capital_loss",   "int", (0, 4356)),
        ("Workclass  [Private / Self-emp-not-inc / Self-emp-inc / Federal-gov / Local-gov / State-gov ...or new value]",
                                      "workclass",      "str", None),
        ("Marital status  [Married-civ-spouse / Never-married / Divorced / Separated / Widowed ...or new value]",
                                      "marital_status", "str", None),
        ("Occupation  [Exec-managerial / Prof-specialty / Adm-clerical / Sales / Other-service ...or new value like AI-engineer]",
                                      "occupation",     "str", None),
        ("Relationship  [Husband / Wife / Own-child / Not-in-family / Unmarried ...or new value]",
                                      "relationship",   "str", None),
        ("Sex  [Male / Female  ...or new value like Non-binary]",
                                      "sex",            "str", None),
    ]

    entry_count         = 0
    pred_0 = pred_1     = 0
    entries_since_check = 0    # tracks how many entries since last drift check

    while True:
        print(f"\n  --- Person #{entry_count + 1} ---")
        record = {}
        stop   = False

        for display, key, dtype, constraint in FIELDS:
            while True:
                raw_val = input(f"  {display}: ").strip()

                if raw_val.lower() == "done":
                    stop = True
                    break

                if not raw_val:
                    print("  Please enter a value.")
                    continue

                if dtype == "int":
                    try:
                        val = int(raw_val)
                    except ValueError:
                        print("  Please enter a whole number.")
                        continue
                    if constraint is not None:
                        lo, hi = constraint
                        if not (lo <= val <= hi):
                            print(f"  Must be between {lo} and {hi}.")
                            continue
                else:
                    val   = raw_val
                    known = list(encoders[key].classes_) if key in encoders else []
                    if val not in known:
                        print(f"  NOTE: '{val}' is a NEW category (not in 1994 training).")
                        print(f"        PSI test will detect this as categorical drift.")

                record[key] = val
                break

            if stop:
                break

        if stop:
            # ── FIX 5: Force final drift check if user stops early ──
            if entries_since_check > 0:
                print(f"\n  Running final drift check on {entries_since_check} entries...")
                _force_drift_check(monitor, FEATURE_COLS)
            print("\n  Stopping.")
            break

        # ── Predict ───────────────────────────────────────────────
        row_df     = pd.DataFrame([record])
        X_input    = prepare_features(row_df)
        prediction = int(model.predict(X_input)[0])
        proba      = float(model.predict_proba(X_input)[0][1])

        label = ">50K  (HIGH earner)" if prediction == 1 else "<=50K"
        print(f"\n  Prediction  : {label}")
        print(f"  Confidence  : {proba*100:.1f}% chance of >50K")

        if prediction == 0: pred_0 += 1
        else:               pred_1 += 1

        monitor.log_prediction(features=record, prediction=prediction)
        entry_count         += 1
        entries_since_check += 1

        print(f"\n  Saved. Total: {entry_count}  |  <=50K: {pred_0}  >50K: {pred_1}")

        if entries_since_check >= 1000:
            entries_since_check = 0
            print()
            print("  +----------------------------------------------------------+")
            print("  |  DRIFT CHECK FIRED (1000 predictions logged)               |")
            print("  |  Dashboard will update in ~30 seconds.                   |")
            print("  |  -> http://localhost:8501                                |")
            print("  +----------------------------------------------------------+")
            print()

        cont = input("\n  Add another person? (yes / no): ").strip().lower()
        if cont not in ("yes", "y"):
           
            if entries_since_check > 0:
                print(f"\n  Running final drift check on {entries_since_check} entries...")
                _force_drift_check(monitor, FEATURE_COLS)
            break

    print(f"\n  Session complete: {entry_count} people entered.")



# ══════════════════════════════════════════════════════════════════
# HELPER: Force a drift check manually (used when user exits early)
# ══════════════════════════════════════════════════════════════════
def _force_drift_check(monitor, feature_cols):
    """Run detect() on whatever is currently in the prediction log."""
    try:
        log_df = monitor.logger.load()
        if log_df.empty:
            print("  No predictions logged yet.")
            return
        missing = [c for c in feature_cols if c not in log_df.columns]
        if missing:
            print(f"  Cannot run check — missing columns: {missing}")
            return
        new_df = log_df[feature_cols].copy()
        result = monitor.detect(new_df)
        health = result["health"]
        score  = round(result["global_score"], 4)
        icons  = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}
        print(f"\n  {icons.get(health,'?')} Health: {health}  |  Global Score: {score}")
        drifted = [r["feature"] for r in result["feature_results"] if r["drift_detected"]]
        if drifted:
            print(f"  Drifted features: {', '.join(drifted)}")
        else:
            print("  No significant drift detected in this batch.")
        print("  Dashboard updated. Check http://localhost:8501")
    except Exception as e:
        print(f"  (Final drift check failed: {e})")


try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    monitor.stop_auto_monitoring()
    print("\n  Stopped. Goodbye!")