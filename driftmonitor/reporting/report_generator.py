from datetime import datetime
import json
import os


def write_report(results, global_drift, health, action, output_path="outputs/drift_report.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("  DATA DRIFT MONITOR REPORT\n")
        f.write(f"  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"  GLOBAL DRIFT SCORE : {round(global_drift, 4)}\n")
        f.write(f"  SYSTEM HEALTH      : {health}\n")
        f.write(f"  RECOMMENDED ACTION : {action}\n\n")

        f.write("-" * 50 + "\n")
        f.write("  FEATURE BREAKDOWN\n")
        f.write("-" * 50 + "\n\n")

        for r in results:
            drift_flag = "DRIFT DETECTED" if r["drift_detected"] else "No drift"
            f.write(f"  Feature  : {r['feature']}\n")
            f.write(f"  Type     : {r['type']}\n")
            f.write(f"  Method   : {r['method']}\n")
            f.write(f"  Score    : {r['score']}\n")
            f.write(f"  Severity : {r['severity']}\n")
            f.write(f"  Status   : {drift_flag}\n")

            if r.get("explanation"):
                f.write("  Reason   :\n")
                for line in r["explanation"]:
                    f.write(f"    - {line}\n")
            f.write("\n")


def append_to_history(result, history_path="outputs/results_history.jsonl"):
   
    os.makedirs(os.path.dirname(history_path), exist_ok=True) if os.path.dirname(history_path) else None

    import numpy as np

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    record = {
        "timestamp": datetime.now().isoformat(),
        **result
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=convert) + "\n")
