import numpy as np


def calculate_psi(expected, actual, buckets=10):

    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints)

    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = 0.0
    for e, a in zip(expected_perc, actual_perc):
        e = max(e, 1e-4)
        a = max(a, 1e-4)
        psi += (a - e) * np.log(a / e)

    return float(psi)


def calculate_categorical_psi(expected_dist, actual_dist):
   
    psi = 0.0
    for e, a in zip(expected_dist, actual_dist):
        e = max(float(e), 1e-4)
        a = max(float(a), 1e-4)
        psi += (a - e) * np.log(a / e)
    return float(psi)
