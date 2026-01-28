# utils/shap_explain.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd


__all__ = [
    "run_kernel_shap", 
    "shap_values_for_model",
    "save_beeswarm_plots"
]



# SHAP is optional; keep the rest of the pipeline working even if missing.
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False


# =========================
# Type & model detectors
# =========================
def _safe_import_xgboost() -> bool:
    try:
        import xgboost as xgb  # noqa: F401
        return True
    except Exception:
        return False

def _is_xgb_model(model) -> bool:
    # Works for sklearn API (XGBClassifier) or Booster-backed wrappers
    name = type(model).__name__.lower()
    return ("xgb" in name) or hasattr(model, "get_booster")

def _is_rf_model(model) -> bool:
    name = type(model).__name__.lower()
    return ("randomforest" in name) or ("extratrees" in name) or ("gradientboosting" in name)

def _is_linear_model(model) -> bool:
    name = type(model).__name__.lower()
    return ("logisticregression" in name) or ("sgdclassifier" in name) or ("ridgeclassifier" in name)

def _is_sklearn_mlp(model) -> bool:
    name = type(model).__name__.lower()
    return "mlpclassifier" in name

def _is_tabnet(model) -> bool:
    # pytorch_tabnet.tab_model.TabNetClassifier
    name = type(model).__name__.lower()
    return ("tabnetclassifier" in name) or ("tabnet" in name)


# =========================
# Data helpers
# =========================
def _default_background(X, max_background=200, random_state=0):
    """
    Small background set for SHAP explainers.
    Accepts numpy array or DataFrame; returns the same type.
    """
    if hasattr(X, "values"):
        arr = X.values
        df_cols = list(X.columns)
    else:
        arr = np.asarray(X)

    if arr.shape[0] > max_background:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(arr.shape[0], size=max_background, replace=False)
        arr = arr[idx]

    if hasattr(X, "values"):
        return pd.DataFrame(arr, columns=df_cols)
    return arr

def _wrap_as_dataframe(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)


# =========================
# Explainer selection
# =========================
def choose_explainer(
    model,
    X_background,
    *,
    model_type_hint: str | None = None,
    prefer_kernel: bool = False,
):
    """
    Pick a SHAP explainer for the given model & background.
    Set prefer_kernel=True to force model-agnostic KernelExplainer.
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap is not installed. Please `pip install shap` first.")

    # Normalize background to numpy for SHAP
    Xb = X_background.values if hasattr(X_background, "values") else np.array(X_background)
    hint = (model_type_hint or "").lower()

    # Force Kernel if requested (works for everything)
    if prefer_kernel:
        return shap.KernelExplainer(_proba_callable(model), shap.sample(Xb, nsamples=len(Xb), random_state=0))

    # Use TreeExplainer where possible (fast & accurate for trees)
    if hint in {"xgb", "xgboost"} or _is_xgb_model(model):
        # For XGBoost, SHAP usually handles predict_proba properly when using TreeExplainer.
        return shap.TreeExplainer(model, data=Xb, model_output="probability")
    if hint in {"rf", "random_forest"} or _is_rf_model(model):
        return shap.TreeExplainer(model, data=Xb, model_output="probability")

    # Linear models (fast)
    if hint in {"lr", "linear", "logreg"} or _is_linear_model(model):
        try:
            # LinearExplainer prefers background (Xb)
            return shap.LinearExplainer(model, Xb)
        except Exception:
            # generic fallback
            return shap.Explainer(_proba_callable(model), Xb)

    # TabNet / MLP / others → generic Explainer (will fall back to Kernel internally)
    if hint in {"tabnet"} or _is_tabnet(model):
        return shap.Explainer(_proba_callable(model), Xb)

    if hint in {"mlp", "nn"} or _is_sklearn_mlp(model):
        try:
            return shap.Explainer(_proba_callable(model), Xb)
        except Exception:
            return shap.KernelExplainer(_proba_callable(model), Xb)

    # Default
    return shap.Explainer(_proba_callable(model), Xb)


def _proba_callable(model):
    """Return a function X -> prob of class 1, robust to numpy/DataFrame input."""
    def f(X):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            # Some wrappers expect DataFrame column names
            if not isinstance(X, pd.DataFrame):
                # Try to infer number of features from model if possible
                n = X.shape[1]
                cols = [f"f{i}" for i in range(n)]
                X = pd.DataFrame(X, columns=cols)
            return model.predict_proba(X)[:, 1]
    return f


# =========================
# SHAP value computation
# =========================
def shap_values_for_model(
    model,
    X_eval,
    *,
    feature_names=None,
    background_X=None,
    model_type_hint: str | None = None,
    max_background: int = 200,
    prefer_kernel: bool = False,
    return_dataframe: bool = True,
):
    """
    Compute SHAP values for a fitted classifier on X_eval.

    Returns:
        shap_vals_signed: (n_samples, n_features) for positive class (1)
        shap_vals_abs   : absolute SHAP
        base_value      : expected model output (positive class)
        df_signed, df_abs: optional DataFrames
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap is not installed. Please `pip install shap` first.")

    X_eval_df = _wrap_as_dataframe(X_eval, feature_names)
    if background_X is None:
        background_X = _default_background(X_eval_df, max_background=max_background, random_state=0)
    background_df = _wrap_as_dataframe(background_X, X_eval_df.columns.tolist())

    explainer = choose_explainer(
        model, background_df,
        model_type_hint=model_type_hint,
        prefer_kernel=prefer_kernel
    )

    # Compute SHAP explanations
    shap_raw = explainer(X_eval_df)  # shap.Explanation

    # Pick the positive class column if multi-output
    values = shap_raw.values
    base_values = shap_raw.base_values

    if values.ndim == 3 and values.shape[2] >= 2:
        # Use last (or 1) as positive class; prefer class index 1 if present.
        pos_idx = 1 if values.shape[2] > 1 else values.shape[2] - 1
        shap_signed = values[:, :, pos_idx]
        if np.ndim(base_values) == 2 and base_values.shape[1] >= 2:
            base_value = float(np.mean(base_values[:, pos_idx]))
        else:
            base_value = float(np.mean(base_values))
    elif values.ndim == 2:
        shap_signed = values
        base_value = float(np.mean(base_values)) if np.ndim(base_values) >= 1 else float(base_values)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    shap_abs = np.abs(shap_signed)

    if return_dataframe:
        cols = list(X_eval_df.columns)
        df_signed = pd.DataFrame(shap_signed, columns=cols, index=X_eval_df.index)
        df_abs    = pd.DataFrame(shap_abs,    columns=cols, index=X_eval_df.index)
        return shap_signed, shap_abs, base_value, df_signed, df_abs

    return shap_signed, shap_abs, base_value, None, None


def shap_feature_importance(
    shap_signed: np.ndarray | pd.DataFrame,
    shap_abs:    np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate SHAP into global feature importance:
    mean(|SHAP|), mean(SHAP), std(|SHAP|).
    """
    if isinstance(shap_abs, pd.DataFrame):
        feature_names = list(shap_abs.columns)
        abs_vals = shap_abs.values
        signed_vals = shap_signed.values if isinstance(shap_signed, pd.DataFrame) else shap_signed
    else:
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(shap_abs.shape[1])]
        abs_vals = shap_abs
        signed_vals = shap_signed

    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(abs_vals, axis=0),
        "mean_signed_shap": np.mean(signed_vals, axis=0),
        "std_abs_shap": np.std(abs_vals, axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return imp


# =========================
# Plot saving helpers
# =========================
def save_beeswarm_plots(
    shap_signed: np.ndarray | pd.DataFrame,
    X_eval: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    out_dir: str = "./shap_outputs",
    prefix: str = "",
) -> tuple[str, str]:
    """
    Saves two beeswarm plots:
      - Signed SHAP
      - Absolute |SHAP|
    Returns (signed_path, abs_path).
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap is not installed. Please `pip install shap` first.")

    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    X_eval_df = _wrap_as_dataframe(X_eval, feature_names)
    values = shap_signed.values if isinstance(shap_signed, pd.DataFrame) else np.asarray(shap_signed)
    feature_names = feature_names or list(X_eval_df.columns)

    # Signed
    plt.figure()
    shap.summary_plot(values, X_eval_df, feature_names=feature_names, show=False, max_display=max_display)
    signed_path = os.path.join(out_dir, f"{prefix}beeswarm_signed.png")
    plt.title("SHAP Beeswarm (signed)")
    plt.tight_layout()
    plt.savefig(signed_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Absolute
    plt.figure()
    shap.summary_plot(np.abs(values), X_eval_df, feature_names=feature_names, show=False, max_display=max_display)
    abs_path = os.path.join(out_dir, f"{prefix}beeswarm_abs.png")
    plt.title("SHAP Beeswarm (|value|)")
    plt.tight_layout()
    plt.savefig(abs_path, dpi=150, bbox_inches="tight")
    plt.close()

    return signed_path, abs_path


# =========================
# One-call Kernel SHAP runner (your requested workflow)
# =========================
def run_kernel_shap(
    model,
    X_val: pd.DataFrame,
    *,
    y_val=None,                 # optional: for stratified subsampling
    feature_names: list[str] | None = None,
    background_size: int = 100,
    sample_size: int = 500,
    max_display: int = 20,
    save_dir: str = "./shap_outputs",
    prefix: str = "",
    model_type_hint: str | None = None,
    seed: int = 0,
    # NEW: CSV saving controls
    save_csv: bool = True,
    global_csv_name: str = "global_shap.csv",
    local_csv_name: str = "local_shap.csv",
) -> dict:
    """
    Runs Kernel SHAP for `model` on a subset of X_val, saves beeswarm plots,
    and (optionally) writes global/local SHAP CSVs.

    Returns a dict with paths and DataFrames:
      - 'signed_plot', 'abs_plot'
      - 'explained_rows' (DataFrame subset)
      - 'shap_signed' (DataFrame, signed local SHAP)
      - 'shap_abs'    (DataFrame, absolute local SHAP)
      - 'base_value'  (float)
      - 'importance'  (DataFrame, global importance)
      - NEW: 'global_csv', 'local_csv' (str paths if saved)
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap is not installed. Please `pip install shap` first.")

    # --- Subsampling helpers
    def _rng(seed): return np.random.default_rng(seed)
    def _strat_idx(y, n, seed):
        y = np.asarray(y).reshape(-1)
        r = _rng(seed)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n = min(n, len(y))
        if len(pos) == 0 or len(neg) == 0:
            return r.choice(len(y), size=n, replace=False)
        n_pos = int(round(n * len(pos) / len(y)))
        n_pos = min(n_pos, len(pos))
        n_neg = n - n_pos
        n_neg = min(n_neg, len(neg))
        if n_pos + n_neg < n:
            rem = n - (n_pos + n_neg)
            take_pos = min(rem, len(pos) - n_pos)
            n_pos += take_pos
            n_neg += (rem - take_pos)
        pick_pos = r.choice(pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
        pick_neg = r.choice(neg, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
        idx = np.concatenate([pick_pos, pick_neg])
        r.shuffle(idx)
        return idx

    X_val_df = _wrap_as_dataframe(X_val, feature_names)
    feature_names = list(X_val_df.columns)

    # Background
    X_bg = _default_background(X_val_df, max_background=background_size, random_state=seed)

    # Rows to explain
    if y_val is None:
        idx = _rng(seed).choice(len(X_val_df), size=min(sample_size, len(X_val_df)), replace=False)
    else:
        idx = _strat_idx(y_val, n=min(sample_size, len(X_val_df)), seed=seed)
    X_explain = X_val_df.iloc[idx]

    # Force Kernel SHAP explainer
    explainer = choose_explainer(
        model, X_bg,
        model_type_hint=model_type_hint,
        prefer_kernel=True
    )
    shap_raw = explainer(X_explain)  # shap.Explanation

    # Select positive class if multi-output
    values = shap_raw.values
    base_values = shap_raw.base_values
    if values.ndim == 3 and values.shape[2] >= 2:
        pos_idx = 1 if values.shape[2] > 1 else values.shape[2] - 1
        shap_signed = values[:, :, pos_idx]
        if np.ndim(base_values) == 2 and base_values.shape[1] >= 2:
            base_value = float(np.mean(base_values[:, pos_idx]))
        else:
            base_value = float(np.mean(base_values))
    elif values.ndim == 2:
        shap_signed = values
        base_value = float(np.mean(base_values)) if np.ndim(base_values) >= 1 else float(base_values)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}")

    shap_abs = np.abs(shap_signed)

    # Global importance table
    imp_df = shap_feature_importance(shap_signed, shap_abs, feature_names=feature_names)

    # Save plots
    os.makedirs(save_dir, exist_ok=True)
    signed_path, abs_path = save_beeswarm_plots(
        shap_signed, X_explain,
        feature_names=feature_names, max_display=max_display,
        out_dir=save_dir, prefix=prefix
    )

    # Local SHAP matrices as DataFrames (keep original row indices)
    df_signed = pd.DataFrame(shap_signed, columns=feature_names, index=X_explain.index)
    df_abs    = pd.DataFrame(shap_abs,    columns=feature_names, index=X_explain.index)

    # === NEW: save CSVs ===
    global_csv_path = None
    local_csv_path  = None
    if save_csv:
        global_csv_path = os.path.join(save_dir, f"{prefix}{global_csv_name}")
        local_csv_path  = os.path.join(save_dir, f"{prefix}{local_csv_name}")

        # Global: ranked by mean_abs_shap
        imp_df.to_csv(global_csv_path, index=False)

        # Local: signed SHAP per-row (wide format). Keep an index column for traceability.
        out_local = df_signed.copy()
        out_local.insert(0, "row_index", out_local.index)
        out_local.to_csv(local_csv_path, index=False)

    return dict(
        signed_plot=signed_path,
        abs_plot=abs_path,
        explained_rows=X_explain,
        shap_signed=df_signed,
        shap_abs=df_abs,
        base_value=base_value,
        importance=imp_df,
        # NEW: file paths
        global_csv=global_csv_path,
        local_csv=local_csv_path,
    )
