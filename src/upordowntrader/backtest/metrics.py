"""
Scores for chances against what happened.

    log loss   the average of -log(the chance given to what happened). Lower is better. A coin flip scores log 2, 0.693, and
               a confident miss costs the most, as it does a trader.
    brier      the average squared gap between the chance and what happened, 1 or 0. A coin flip scores 0.25.
"""

import numpy as np
from itertools import pairwise

CLIP = 1e-6     # Chances are kept this far from 0 and 1, so one confident miss costs a lot but not without limit.
# Bands of the favorite's chance for favorites(), finer toward certainty, where a small miscalibration costs the most.
FAVORITE_EDGES = (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)


def log_loss(p, up):
    """
    The log loss of chances of Up against whether Up won.
    """
    p = np.clip(p, CLIP, 1 - CLIP)
    return float(-np.mean(np.where(up, np.log(p), np.log1p(-p))))


def brier(p, up):
    """
    The Brier score of chances of Up against whether Up won.
    """
    return float(np.mean((p - up) ** 2))


def favorites(p, up, windows, edges=FAVORITE_EDGES):
    """
    Calibration by the favorite's chance, the larger of Up's and Down's, folded together since the model treats them
    alike. For each band of edges with evaluations in it: how many, from how many windows, the average chance the model
    gave the favorite, how often the favorite won, and that rate's standard error, counting each window once, since the
    evaluations of one window share its outcome.
    """
    favorite = np.maximum(p, 1 - p)
    won = np.where(p >= 0.5, up, ~up)
    rows = []
    for low, high in pairwise(edges):
        inside = (favorite >= low) & ((favorite < high) | (high == edges[-1]))
        if inside.any():
            rate, n_windows = won[inside].mean(), len(np.unique(windows[inside]))
            rows.append({"low": low, "high": high, "evaluations": int(inside.sum()), "windows": n_windows, "said": float(favorite[inside].mean()),
                         "won": float(rate), "error": float(np.sqrt(rate * (1 - rate) / n_windows))})
    return rows
