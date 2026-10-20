"""dtaidistance.dtw_ndim.distance without a window (the only call WaveLock
makes: gesture_compare.compute_dtw_distance).

Same definition as the C implementation: the cost of matching frame i of s1
with frame j of s2 is their squared Euclidean distance, costs accumulate
along the cheapest monotone path, and the result is the square root of the
total. The recurrence is evaluated one anti-diagonal at a time so each step
is a vector operation.
"""
import numpy as np


def distance(s1, s2, window=None, **kwargs):
    if window is not None or kwargs:
        raise NotImplementedError("the mobile DTW supports distance(s1, s2) only")
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s2, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)          # (n, m)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for k in range(2, n + m + 1):                                       # i + j = k
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        best = np.minimum(np.minimum(acc[i - 1, j - 1], acc[i - 1, j]), acc[i, j - 1])
        acc[i, j] = cost[i - 1, j - 1] + best
    return float(np.sqrt(acc[n, m]))
