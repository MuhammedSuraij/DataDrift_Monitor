from driftmonitor.engine.common_engine import analyze_feature_drift, system_health, recommended_action
from driftmonitor.drift.global_drift import compute_global_drift


def run_stream_drift(train_df, new_df, weights=None):
    
    if weights is None:
        weights = {}

    print("\n--- STREAM DRIFT CHECK STARTED ---")

    common_cols = [col for col in train_df.columns if col in new_df.columns]

    feature_scores = {}
    results = []

    for col in common_cols:
        try:
            res = analyze_feature_drift(train_df, new_df, col)
            feature_scores[col] = res["score"]
            results.append(res)
            print(
                f"  {col:20s} | {res['method']:20s} | "
                f"Drift={str(res['drift_detected']):5s} | "
                f"Severity={res['severity']:6s} | Score={res['score']:.4f}"
            )
        except Exception as e:
            print(f"  [WARN] Could not analyze column '{col}': {e}")

    global_score = compute_global_drift(feature_scores, weights)
    health = system_health(global_score)
    action = recommended_action(health)

    print(f"\n  GLOBAL DRIFT SCORE : {round(global_score, 4)}")
    print(f"  SYSTEM HEALTH      : {health}")
    print(f"  ACTION             : {action}")
    print("--- DRIFT CHECK COMPLETE ---\n")

    return results, global_score, health, action
