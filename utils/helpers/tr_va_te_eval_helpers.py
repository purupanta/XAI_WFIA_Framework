# utils/helpers/tr_va_te_eval_helpers.py

__all__ = [
    "save_eval_table_csv",
    "save_probs_csv",
    "save_thresholds_csv",
    "evaluate_on_val", 
    "evaluate_on_test"
]

# ======= SAVE TO CSV FUNCTIONS =======

# ======= CSV save helpers (tiny) =======
from pathlib import Path
import pandas as pd
import numpy as np

def _ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_eval_table_csv(df: pd.DataFrame, path: str | Path, *, verbose: int = 0) -> Path:
    """Save a metrics table (the DataFrame returned by evaluate_*) to CSV."""
    p = _ensure_dir(Path(path))
    out_df = (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
    out_df.to_csv(p, index=True)
    if verbose > 0:
        r, c = out_df.shape if hasattr(out_df, "shape") else (0, 0)
        print(f"✓ Saved metrics table: {p.resolve()}  [{r}×{c}]")
    return p

def save_probs_csv(probs: dict[str, np.ndarray], path: str | Path, *, verbose: int = 0) -> Path:
    """Save per-model probability vectors to one CSV (columns = models)."""
    p = _ensure_dir(Path(path))
    if not probs:
        pd.DataFrame().to_csv(p, index=False)
        if verbose > 0:
            print(f"✓ Saved probs (empty): {p.resolve()}")
        return p
    # Align lengths by the longest (shorter ones padded with NaN)
    max_len = max(len(v) for v in probs.values())
    data = {k: np.asarray(v, dtype=float) for k, v in probs.items()}
    for k, v in data.items():
        if len(v) < max_len:
            pad = np.full(max_len - len(v), np.nan)
            data[k] = np.concatenate([v, pad])
    df = pd.DataFrame(data)
    df.to_csv(p, index=False)
    if verbose > 0:
        print(f"✓ Saved probs: {p.resolve()}  [rows={len(df)}, models={len(df.columns)}]")
    return p

def save_thresholds_csv(thresholds: dict[str, float], path: str | Path, *, verbose: int = 0) -> Path:
    """Save {model: threshold} to CSV."""
    p = _ensure_dir(Path(path))
    s = pd.Series(thresholds, name="threshold")
    s.to_csv(p, header=True)
    if verbose > 0:
        print(f"✓ Saved thresholds: {p.resolve()}  [models={len(s)}]")
    return p

def save_preds_csv(y_preds: dict[str, np.ndarray], path: str | Path, *, verbose: int = 0) -> Path:
    """Save per-model hard predictions to one CSV (columns = models)."""
    p = _ensure_dir(Path(path))
    if not y_preds:
        pd.DataFrame().to_csv(p, index=False)
        if verbose > 0:
            print(f"✓ Saved predictions (empty): {p.resolve()}")
        return p
    max_len = max(len(v) for v in y_preds.values())
    data = {k: np.asarray(v, dtype=float) for k, v in y_preds.items()}
    for k, v in data.items():
        if len(v) < max_len:
            pad = np.full(max_len - len(v), np.nan)
            data[k] = np.concatenate([v, pad])
    df = pd.DataFrame(data)
    df.to_csv(p, index=False)
    if verbose > 0:
        print(f"✓ Saved predictions: {p.resolve()}  [rows={len(df)}, models={len(df.columns)}]")
    return p


# ======= DISPLAY FUNCTIONS =======

# ======= Evaluation helpers (compact, robust) =======
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    accuracy_score, f1_score
)
from math import isfinite

# --- calibration ---
def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    # clamp to [0,1] to be safe
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.nanmean(y_prob[mask]))
        acc  = float(np.nanmean(y_true[mask]))
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

# --- proba dispatcher (TabNet-safe with fallbacks) ---
def proba(model, X):
    """Return positive-class probabilities for binary problems."""
    # TabNet special-case
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        if isinstance(model, TabNetClassifier):
            Xv = X.values if hasattr(X, "values") else X
            return np.asarray(model.predict_proba(Xv))[:, 1]
    except Exception:
        pass

    # Generic: prefer predict_proba
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X)
        P = np.asarray(P)
        # handle shape (n,2) or (n,)
        if P.ndim == 2 and P.shape[1] >= 2:
            return P[:, 1]
        elif P.ndim == 1:
            return P
    # Fallback: decision_function -> sigmoid/minmax to [0,1]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        # numeric guard
        if np.all(np.isfinite(s)):
            # logistic mapping is safer than minmax for outliers
            return 1.0 / (1.0 + np.exp(-s))
        # last resort: min-max
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        if s_max > s_min:
            return (s - s_min) / (s_max - s_min)
        return np.full_like(s, 0.5, dtype=float)
    raise RuntimeError("Model does not expose predict_proba or decision_function.")

# --- threshold selection ---
def choose_threshold(y_true, p, threshold):
    if str(threshold).lower() == "auto":
        p = np.asarray(p, dtype=float)
        # guard for degenerate arrays
        if np.allclose(p, p[0]):
            return 0.5
        ts = np.linspace(0.0, 1.0, 1001)
        # zero_division=0 avoids warnings when a class is missing
        f1s = np.array([f1_score(y_true, p >= t, zero_division=0) for t in ts])
        return float(ts[int(np.argmax(f1s))])
    return float(threshold)

# --- metric computation for one model ---
def compute_metrics(y_true, p, thr, sample_weight=None):
    out = {}
    p = np.asarray(p, dtype=float)
    try: out["roc_auc"] = roc_auc_score(y_true, p, sample_weight=sample_weight)
    except Exception: pass
    try: out["pr_auc"]  = average_precision_score(y_true, p, sample_weight=sample_weight)
    except Exception: pass
    try: out["ece"]     = expected_calibration_error(y_true, p, n_bins=15)
    except Exception: pass
    try: out["logloss"] = log_loss(y_true, p, sample_weight=sample_weight)
    except Exception: pass
    try:
        y_pred = (p >= thr).astype(int)
        out["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    except Exception: pass
    return out

# --- which X (scaled vs raw) based on model name ---
_SCALED_MODELS = {"lr", "mlp"}
def pick_features_for_model(name, X_raw, X_scaled, scaled_models=_SCALED_MODELS):
    return X_scaled if name in scaled_models else X_raw

# --- split fetcher (supports optional sample weights) ---
def _get_split_data(result, split="val"):
    if split == "val":
        X, Xs, y, w = result["X_val"], result["X_val_scaled"], result["y_val"], result.get("w_val")
    elif split == "test":
        X, Xs, y, w = result["X_test"], result["X_test_scaled"], result["y_test"], result.get("w_test")
    else:
        raise ValueError("split must be 'val' or 'test'")
    y = y.values if hasattr(y, "values") else y
    if w is not None:
        w = w.values if hasattr(w, "values") else w
    return X, Xs, y, w

# --- generic evaluator used by val/test wrappers ---
def _evaluate(split, result, models, threshold=0.5, verbose=True, return_preds=False, scaled_models=_SCALED_MODELS):
    X_raw, X_scaled, y_true, w = _get_split_data(result, split=split)
    rows, probs, thresholds, y_preds = [], {}, {}, {}

    for name, mdl in models.items():
        try:
            X_for_model = pick_features_for_model(name, X_raw, X_scaled, scaled_models)
            p = proba(mdl, X_for_model)
            probs[name] = p
            thr = choose_threshold(y_true, p, threshold)
            thresholds[name] = thr
            m = {"model": name, "threshold": thr}
            m.update(compute_metrics(y_true, p, thr, sample_weight=w))
            rows.append(m)
            if return_preds:
                y_preds[name] = (np.asarray(p) >= thr).astype(int)
        except Exception as e:
            print(f"[WARN] {split}:{name}: could not evaluate -> {e}")

    if not rows:
        print(f"[ERROR] No models produced metrics on {split}.")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows).set_index("model")
        # Nice default ordering
        by = "roc_auc" if "roc_auc" in df.columns else None
        if by:
            df = df.sort_values(by, ascending=False)

    if verbose and not df.empty:
        print(f"\n {split.capitalize()} performance:")
        cols = [c for c in ["roc_auc","pr_auc","ece","logloss","accuracy","threshold"] if c in df.columns]
        print(df[cols].round(3))

    return (df, probs, thresholds, y_preds) if return_preds else (df, probs, thresholds)

# --- public wrappers ---
def evaluate_on_val(result, models, threshold=0.5, verbose=True, return_preds=False, scaled_models=_SCALED_MODELS):
    return _evaluate("val", result, models, threshold, verbose, return_preds, scaled_models)

def evaluate_on_test(result, models, threshold=0.5, verbose=True, return_preds=False, scaled_models=_SCALED_MODELS):
    # returns thresholds too for symmetry (useful if you pass in 'auto')
    return _evaluate("test", result, models, threshold, verbose, return_preds, scaled_models)

# --- helper: optimize on val, then apply those thresholds to test ---
def evaluate_test_with_val_thresholds(result, models, verbose=True, return_preds=False, scaled_models=_SCALED_MODELS):
    """1) Find F1-optimal thresholds on validation; 2) Evaluate test using those thresholds."""
    val_df, _, val_thr = evaluate_on_val(result, models, threshold="auto", verbose=verbose, return_preds=False, scaled_models=scaled_models)
    # Evaluate test per model using its validation threshold
    test_rows, test_probs, test_thr, y_preds = [], {}, {}, {}
    X_raw, X_scaled, y_true, w = _get_split_data(result, split="test")

    for name, mdl in models.items():
        if name not in val_thr:
            continue
        try:
            X_for_model = pick_features_for_model(name, X_raw, X_scaled, scaled_models)
            p = proba(mdl, X_for_model)
            test_probs[name] = p
            thr = float(val_thr[name])
            test_thr[name] = thr
            m = {"model": name, "threshold": thr}
            m.update(compute_metrics(y_true, p, thr, sample_weight=w))
            test_rows.append(m)
            if return_preds:
                y_preds[name] = (np.asarray(p) >= thr).astype(int)
        except Exception as e:
            print(f"[WARN] test:{name}: could not evaluate -> {e}")

    if not test_rows:
        print("[ERROR] No models produced metrics on test.")
        test_df = pd.DataFrame()
    else:
        test_df = pd.DataFrame(test_rows).set_index("model")
        if "roc_auc" in test_df.columns:
            test_df = test_df.sort_values("roc_auc", ascending=False)

    if verbose and not test_df.empty:
        print("Test performance (val-optimized thresholds):")
        cols = [c for c in ["roc_auc","pr_auc","ece","logloss","accuracy","threshold"] if c in test_df.columns]
        print(test_df[cols].round(3))

    return (val_df, test_df, test_probs, val_thr, test_thr, y_preds) if return_preds else (val_df, test_df, test_probs, val_thr, test_thr)
