def compute_global_drift(feature_scores, feature_weights=None):
    
    if feature_weights is None:
        feature_weights = {}

    total = 0.0
    weight_sum = 0.0

    for feature, score in feature_scores.items():
        weight = feature_weights.get(feature, 1)
        total += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return total / weight_sum
