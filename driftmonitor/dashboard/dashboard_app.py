import streamlit as st
import json
import os
import shutil
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Drift Monitor", layout="wide", page_icon="📊")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="drift_refresh")
except ImportError:
    pass

RESULT_FILE  = "outputs/latest_results.json"
HISTORY_FILE = "outputs/results_history.jsonl"
PRED_LOG     = "outputs/prediction_log.csv"
OUTPUT_DIR   = "outputs"

# ── Header + Reset Button ─────────────────────────────────────────
col_title, col_reset = st.columns([5, 1])
with col_title:
    st.title("📊 Data Drift Monitoring Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col_reset:
    st.write("")   # vertical spacer
    st.write("")
    if st.button("🗑️ Reset All", type="secondary", help="Delete all outputs and start fresh"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        st.success("All outputs cleared. Run your script again to start fresh.")
        st.rerun()

st.divider()

tab_latest, tab_history, tab_pred, tab_plots = st.tabs(
    ["Latest Check", "History", "Prediction Log", "Drift Plots"]
)


# ══════════════════════════════════════════════════════════════════
# TAB 1 — LATEST CHECK
# ══════════════════════════════════════════════════════════════════
with tab_latest:
    if not os.path.exists(RESULT_FILE):
        st.warning("No monitoring results yet. Run your example script first.")
        st.stop()

    with open(RESULT_FILE) as f:
        data = json.load(f)

    feature_results = data["feature_results"]
    global_score    = data["global_score"]
    health          = data["health"]
    action          = data["action"]

    color_map = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}
    st.header(f"{color_map.get(health, '')} System Health: {health}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Global Drift Score", round(global_score, 4))
    col2.metric("System Health",      health)
    col3.metric("Recommended Action", action)

    st.divider()
    st.subheader("Feature Drift Analysis")

    rows = []
    for r in feature_results:
        explanation_text = ""
        if r["drift_detected"] and r.get("explanation"):
            explanation_text = "  |  ".join(r["explanation"])

        rows.append({
            "Feature":        r["feature"],
            "Type":           r["type"],
            "Method":         r["method"],
            "Score":          round(r["score"], 4),
            "Severity":       r["severity"],
            "Drift":          "YES" if r["drift_detected"] else "—",
            "Reason":         explanation_text,
        })

    df_results = pd.DataFrame(rows)

    def severity_style(val):
        if val == "HIGH":
            return "background-color: #ffcccc; color: #8b0000"
        elif val == "MEDIUM":
            return "background-color: #fff3cd; color: #856404"
        return ""

    def drift_style(val):
        if val == "YES":
            return "font-weight: bold; color: #c0392b"
        return "color: #888"

    styled = (df_results.style
              .map(severity_style, subset=["Severity"])
              .map(drift_style,    subset=["Drift"]))

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Drifted features summary
    drifted = [r["feature"] for r in feature_results if r["drift_detected"]]
    if drifted:
        st.error(f"**Drifted features:** {', '.join(drifted)}")
    else:
        st.success("No drift detected in this batch.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Global Drift Score Over Time")

    if not os.path.exists(HISTORY_FILE):
        st.info("No history yet. History is written after each drift check.")
    else:
        records = []
        with open(HISTORY_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if records:
            hist_df = pd.DataFrame([
                {
                    "timestamp":    r.get("timestamp", ""),
                    "global_score": r.get("global_score", 0),
                    "health":       r.get("health", ""),
                    "action":       r.get("action", ""),
                }
                for r in records
            ])
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
            hist_df = hist_df.sort_values("timestamp")

            # Colour-coded health column
            st.line_chart(hist_df.set_index("timestamp")["global_score"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("History file is empty.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION LOG
# ══════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Prediction Log")

    if not os.path.exists(PRED_LOG):
        st.info("No predictions logged yet.")
    else:
        # Read safely — handle corrupt/malformed CSV gracefully
        try:
            pred_df = pd.read_csv(PRED_LOG)
        except Exception as e:
            st.error(f"Could not read prediction log: {e}")
            st.info("The log file may be from a previous run with different columns. "
                    "Click 'Reset All' above to clear it.")
            st.stop()

        st.metric("Total Predictions Logged", len(pred_df))

        if "prediction" in pred_df.columns:
            st.subheader("Prediction Distribution")
            st.bar_chart(pred_df["prediction"].value_counts().sort_index())

        st.subheader("Recent Predictions (last 100)")
        st.dataframe(pred_df.tail(100), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — DRIFT PLOTS
# ══════════════════════════════════════════════════════════════════
with tab_plots:
    st.subheader("Drift Visualizations")

    if not os.path.exists(OUTPUT_DIR):
        st.info("No plots yet. Run your example script first.")
    else:
        images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

        if not images:
            st.info("No drift plots yet.")
        else:
            pred_plot  = [i for i in images if "prediction" in i]
            feat_plots = [i for i in images if "prediction" not in i]

            if pred_plot:
                st.markdown("#### Prediction Drift")
                st.image(os.path.join(OUTPUT_DIR, pred_plot[0]),
                         use_container_width=True)
                st.divider()

            st.markdown("#### Feature Drift Plots")
            # Show two plots per row
            for idx in range(0, len(feat_plots), 2):
                c1, c2 = st.columns(2)
                with c1:
                    img = feat_plots[idx]
                    st.caption(img.replace("_", " ").replace(".png", ""))
                    st.image(os.path.join(OUTPUT_DIR, img), use_container_width=True)
                if idx + 1 < len(feat_plots):
                    with c2:
                        img = feat_plots[idx + 1]
                        st.caption(img.replace("_", " ").replace(".png", ""))
                        st.image(os.path.join(OUTPUT_DIR, img), use_container_width=True)