"""Data loading and windowing for the Hungarian Chickenpox benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "hungary_chickenpox.csv"


def load_chickenpox(path=None):
    """Weekly chickenpox cases, one column per county (the Date column is dropped)."""
    df = pd.read_csv(path or DEFAULT_CSV)
    return df.drop(columns=[c for c in df.columns if c.lower() == "date"])


def make_windows(series, window):
    """Rolling windows of a 1-D series: row t is series[t : t + window]. The last column is the forecast target."""
    a = np.asarray(series, dtype=float).ravel()
    return np.lib.stride_tricks.sliding_window_view(a, window).copy()


def split_windows(series, window=8, train_size=400):
    """Chronological split of the rolling windows into (train, test), each of shape (n, window)."""
    W = make_windows(series, window)
    return W[:train_size], W[train_size:]


def split_xy(W):
    """Window matrix -> (inputs, target): the first window-1 columns predict the last one."""
    return W[:, :-1], W[:, -1]
