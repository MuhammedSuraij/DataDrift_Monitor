"""
automated_monitor.py
====================
Background thread that runs drift detection automatically.
"""

import threading
import time
from datetime import datetime

from driftmonitor.alerting.alerting import AlertManager


class AutomatedDriftMonitor:

    def __init__(
        self,
        monitor,
        check_every_n_records=100,
        interval_seconds=None,
        slack_webhook_url=None,
        alert_on=("CRITICAL", "WARNING"),
        feature_columns=None,
        generate_report=True,
        visualize=True,
    ):
        self.monitor               = monitor
        self.check_every_n_records = check_every_n_records
        self.interval_seconds      = interval_seconds
        self.alert_manager         = AlertManager(slack_webhook_url, alert_on)
        self.feature_columns       = feature_columns
        self.generate_report       = generate_report
        self.visualize             = visualize

        self._stop_event           = threading.Event()
        self._thread               = None
        self._last_checked_count   = 0

        if check_every_n_records is None and interval_seconds is None:
            raise ValueError("Set either check_every_n_records or interval_seconds.")

    def start(self):
        print("[AutoMonitor] Starting background drift monitoring...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[AutoMonitor] Running. Call .stop() to halt.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        print("[AutoMonitor] Stopped.")

    def _run_loop(self):
        while not self._stop_event.is_set():
            triggered = False

            if self.interval_seconds is not None:
                triggered = True
                time.sleep(self.interval_seconds)

            elif self.check_every_n_records is not None:
                current_count = self.monitor.logger.count()
                new_records = current_count - self._last_checked_count
                if new_records >= self.check_every_n_records:
                    triggered = True
                else:
                    time.sleep(5)

            if triggered and not self._stop_event.is_set():
                self._run_check()

    def _run_check(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[AutoMonitor] Running drift check at {timestamp}")

        try:
            log_df = self.monitor.logger.load()

            if log_df.empty:
                print("[AutoMonitor] No prediction data yet — skipping.")
                return

            # ── Validate columns match expected feature columns ──────────
            non_feature = {"timestamp", "prediction"}
            if self.feature_columns:
                feature_cols = self.feature_columns
            else:
                feature_cols = [c for c in log_df.columns if c not in non_feature]

            # Safety check: if the log doesn't have our feature columns,
            # it's stale from a previous run — skip silently
            missing = [c for c in feature_cols if c not in log_df.columns]
            if missing:
                print(f"[AutoMonitor] Log columns don't match feature columns "
                      f"(stale file?) — skipping. Missing: {missing[:3]}...")
                self._last_checked_count = 0   # reset so next batch triggers fresh check
                return

            new_df = log_df[feature_cols].copy()

            # Align dtypes with training data
            for col in new_df.columns:
                if col in self.monitor.train_df.columns:
                    try:
                        new_df[col] = new_df[col].astype(self.monitor.train_df[col].dtype)
                    except Exception:
                        pass

            result = self.monitor.detect(new_df)
            self._last_checked_count = self.monitor.logger.count()

            self.alert_manager.send(
                health=result["health"],
                global_score=result["global_score"],
                results=result["feature_results"],
                timestamp=timestamp,
            )

            if "prediction" in log_df.columns and self.monitor.train_predictions is not None:
                self._check_prediction_drift(log_df["prediction"].values)

            if self.generate_report:
                report_path = f"outputs/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                self.monitor.generate_report(
                    result["feature_results"],
                    result["global_score"],
                    result["health"],
                    result["action"],
                    path=report_path,
                )

            if self.visualize:
                self.monitor.visualize(new_df)

        except Exception as e:
            print(f"[AutoMonitor] ERROR during drift check: {e}")
            import traceback
            traceback.print_exc()

    def _check_prediction_drift(self, new_predictions):
        if self.monitor.train_predictions is None:
            return
        from driftmonitor.visualization.drift_visualizer import plot_prediction_drift
        plot_prediction_drift(
            self.monitor.train_predictions,
            new_predictions,
            output_dir="outputs"
        )
        print("[AutoMonitor] Prediction drift plot updated.")