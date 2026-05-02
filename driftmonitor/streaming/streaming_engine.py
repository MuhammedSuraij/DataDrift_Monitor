import pandas as pd
import os
import csv
import threading
from datetime import datetime


class StreamBuffer:
    

    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        self.buffer = []

    def add(self, record):
        self.buffer.append(record)
        return len(self.buffer) >= self.batch_size

    def get_dataframe(self):
        df = pd.DataFrame(self.buffer)
        self.buffer = []
        return df

    def size(self):
        return len(self.buffer)


class PredictionLogger:
    
    def __init__(self, log_path="outputs/prediction_log.csv"):
        self.log_path = log_path
        self._lock = threading.Lock()          # prevents simultaneous writes
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._fieldnames = None                # learned from first log() call

    def reset(self):
        """Delete the log file so a fresh run starts clean."""
        with self._lock:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
            self._fieldnames = None

    def log(self, features: dict, prediction, timestamp=None):
        """Append one prediction row to the CSV (thread-safe)."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        row = {"timestamp": timestamp, **features, "prediction": prediction}

        with self._lock:
            file_exists = os.path.exists(self.log_path)

            if self._fieldnames is None:
                self._fieldnames = list(row.keys())

            if file_exists and self._fieldnames:
                try:
                    existing_cols = pd.read_csv(self.log_path, nrows=0).columns.tolist()
                    if existing_cols != self._fieldnames:
                        os.remove(self.log_path)
                        file_exists = False
                except Exception:
                    os.remove(self.log_path)
                    file_exists = False

            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self._fieldnames,
                    quoting=csv.QUOTE_ALL,     # always quote — prevents comma-in-value bugs
                    extrasaction="ignore",
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def load(self):
        """Load the full prediction log as a DataFrame."""
        with self._lock:
            if not os.path.exists(self.log_path):
                return pd.DataFrame()
            try:
                return pd.read_csv(self.log_path)
            except Exception:
                return pd.DataFrame()

    def load_last_n(self, n):
        df = self.load()
        return df.tail(n).reset_index(drop=True)

    def count(self):
        df = self.load()
        return len(df)