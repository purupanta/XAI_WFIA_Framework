# utils/helpers/logging_helpers.py

__all__ = [
    "Timer", "to_df", "to_series"
]

# -----------------------
# Logging helpers
# -----------------------
import warnings, os, datetime, time
import pandas as pd
def stamp(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self):
        stamp(f"START: {self.label}"); self.t0 = time.time()
        return self
    def __exit__(self, *args):
        stamp(f"END:   {self.label}  (took {time.time()-self.t0:.2f}s)\n")

# -----------------------
# Utils
# -----------------------
def to_df(X, columns):
    if isinstance(X, pd.DataFrame): return X.copy()
    X = np.asarray(X); columns = columns or [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)

def to_series(y, name="target"):
    if isinstance(y, pd.Series): return y.copy()
    return pd.Series(np.asarray(y).ravel(), name=name)
