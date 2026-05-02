import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from driftmonitor.baseline.baseline import compute_baseline, save_baseline, load_baseline
from driftmonitor.engine.main_drift_runner import run_stream_drift
from driftmonitor.reporting.report_generator import write_report, append_to_history
from driftmonitor.streaming.streaming_engine import StreamBuffer, PredictionLogger
from driftmonitor.visualization.drift_visualizer import (
    plot_categorical_drift,
    plot_numeric_drift,
    plot_prediction_drift,
)
from driftmonitor.alerting.alerting import AlertManager


class DriftMonitor:

    def __init__(
        self,
        batch_size=50,
        weights=None,
        slack_webhook_url=None,
        alert_on=("CRITICAL", "WARNING"),
        prediction_log_path="outputs/prediction_log.csv",
    ):

        self.train_df = None
        self.baseline = None
        self.train_predictions = None
        self.weights = weights or {}

        self.buffer = StreamBuffer(batch_size)
        self.logger = PredictionLogger(prediction_log_path)
        self.alert_manager = AlertManager(slack_webhook_url, alert_on)

        self._auto_monitor = None


    def fit(self, train_df, train_predictions=None):

        self.train_df = train_df.copy()
        self.baseline = compute_baseline(train_df)
        self.train_predictions = (
            list(train_predictions) if train_predictions is not None else None
        )
        print(
            f"[DriftMonitor] Baseline set on {len(train_df)} rows, "
            f"{len(train_df.columns)} features."
        )

    def fit_from_baseline(self, baseline_path, reference_df):

        self.baseline = load_baseline(baseline_path)
        self.train_df = reference_df.copy()
        print(f"[DriftMonitor] Baseline loaded from: {baseline_path}")

    def save_baseline(self, path="outputs/baseline.json"):
        if self.baseline is None:
            raise ValueError("Call fit() before save_baseline().")
        save_baseline(self.baseline, path)



    def log_prediction(self, features: dict, prediction):

        self.logger.log(features, prediction)

        ready = self.buffer.add(features)
        if ready:
            batch = self.buffer.get_dataframe()
            print(f"[DriftMonitor] Buffer full ({self.buffer.batch_size} records). Running detection...")
            result = self.detect(batch)
            self.alert_manager.send(
                health=result["health"],
                global_score=result["global_score"],
                results=result["feature_results"],
            )



    def detect(self, new_df):

        if self.train_df is None:
            raise ValueError("You must call fit() first.")

        results, global_score, health, action = run_stream_drift(
            self.train_df, new_df, weights=self.weights
        )

        result = {
            "feature_results": results,
            "global_score": global_score,
            "health": health,
            "action": action,
        }

        self._save_latest(result)
        append_to_history(result)

        return result



    def visualize(self, new_df, output_dir="outputs"):
        
        if self.train_df is None:
            raise ValueError("Call fit() before visualize().")

        for col in self.train_df.columns:
            if col not in new_df.columns:
                continue
            if self.train_df[col].dtype in ["int64", "float64"]:
                plot_numeric_drift(self.train_df[col], new_df[col], col, output_dir)
            else:
                plot_categorical_drift(self.train_df[col], new_df[col], col, output_dir)

        print(f"[DriftMonitor] Plots saved to: {output_dir}/")

    def visualize_predictions(self, new_predictions, output_dir="outputs"):
        """Plot prediction drift. Requires fit(train_predictions=...) to have been called."""
        if self.train_predictions is None:
            print("[DriftMonitor] No training predictions stored. "
                  "Pass train_predictions= to fit().")
            return
        plot_prediction_drift(self.train_predictions, new_predictions, output_dir)
        print(f"[DriftMonitor] Prediction drift plot saved to: {output_dir}/")



    def generate_report(self, results, global_score, health, action,
                        path="outputs/drift_report.txt"):
        write_report(results, global_score, health, action, path)
        print(f"[DriftMonitor] Report saved to: {path}")



    def start_auto_monitoring(
        self,
        check_every_n_records=None,
        interval_seconds=None,
        feature_columns=None,
        generate_report=True,
        visualize=True,
    ):
      
        from driftmonitor.automation.automated_monitor import AutomatedDriftMonitor

        if self._auto_monitor is not None:
            print("[DriftMonitor] Auto-monitoring already running. Call stop_auto_monitoring() first.")
            return

        self._auto_monitor = AutomatedDriftMonitor(
            monitor=self,
            check_every_n_records=check_every_n_records,
            interval_seconds=interval_seconds,
            slack_webhook_url=self.alert_manager.slack_webhook_url,
            alert_on=self.alert_manager.alert_on,
            feature_columns=feature_columns,
            generate_report=generate_report,
            visualize=visualize,
        )
        self._auto_monitor.start()

    def stop_auto_monitoring(self):
        """Stop the background monitoring thread."""
        if self._auto_monitor:
            self._auto_monitor.stop()
            self._auto_monitor = None


    def reset_outputs(self):
       
        import shutil
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
        os.makedirs("outputs", exist_ok=True)
        self.logger.reset()
        print("[DriftMonitor] Outputs cleared — fresh start.")

    def launch_dashboard(self):
        
        dashboard_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dashboard", "dashboard_app.py"
        )

        import importlib.util
        if importlib.util.find_spec("streamlit") is None:
            print("[DriftMonitor] Streamlit not installed.")
            print("  Run:  pip install streamlit streamlit-autorefresh")
            return

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                dashboard_path,
            ],
            stdout=subprocess.DEVNULL,   
            stderr=subprocess.DEVNULL,   
        )
        print("[DriftMonitor] Dashboard launching...")
        print("  URL  : http://localhost:8501")
        print("  Note : browser should open automatically in a few seconds.")
        print("  Note : Streamlit logs are hidden — your terminal stays clean.")
        

    def monitor(self, new_df):
        result = self.detect(new_df)
        self.launch_dashboard()
        return result

    

    def _save_latest(self, results):
        os.makedirs("outputs", exist_ok=True)

        def convert(obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return str(obj)

        with open("outputs/latest_results.json", "w") as f:
            json.dump(results, f, default=convert, indent=2)