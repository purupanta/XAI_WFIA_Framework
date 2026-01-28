# utils/shap_utils.py
import os
import numpy as np
import pandas as pd
import shap

# ================= path helpers =================
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _shap_dir(CONFIGS: dict) -> str:
    d = CONFIGS.get("DIR_tr_va_te_metric_shap_SAVE_DIR")
    if not d:
        raise ValueError("CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] is required.")
    return _ensure_dir(d)

def _model_dir(base: str, model_key: str) -> str:
    return _ensure_dir(os.path.join(base, model_key))

def _run_ts(CONFIGS: dict) -> str:
    ts = str(CONFIGS.get("RUN_TS", "")).strip()
    return f"_{ts}" if ts else ""

def _normpath(p: str) -> str:
    # Pretty path for printing (works on all OSes)
    return os.path.normpath(os.path.abspath(p))

# Pretty labels for prints
_MODEL_LABELS = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "Neural Network (MLP)",
    "xlstm": "XLSTM",
    "tabnet": "TabNet",
}

# ================= SHAP helpers =================
def _bg_masker(X, k: int = 200, seed: int = 0):
    if isinstance(X, pd.DataFrame):
        bg = X.sample(min(len(X), k), random_state=seed)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=min(len(X), k), replace=False)
        bg = X[idx]
    return shap.maskers.Independent(bg)

def _is_pipeline(obj) -> bool:
    name = getattr(obj, "__class__", type(obj)).__name__.lower()
    return ("pipeline" in name) or ("columntransformer" in name)

def _extract_model(m, depth: int = 0, seen=None):
    """Recursively unwrap dicts/pipelines to find an estimator or callable."""
    if seen is None:
        seen = set()
    if id(m) in seen or depth > 7:
        return m
    seen.add(id(m))

    # direct wins
    if hasattr(m, "predict") or hasattr(m, "predict_proba") or callable(m):
        return m

    # dict-like wrappers (project-specific priorities first)
    if isinstance(m, dict):
        priority = (
            "lr_raw", "lr_cal",
            "rf_raw", "rf_cal",
            "xgb_raw", "xgb_cal",
            "mlp_raw", "mlp_cal",
            "xlstm_model", "xlstm_cal",
        )
        for k in priority:
            if k in m:
                res = _extract_model(m[k], depth + 1, seen)
                if hasattr(res, "predict") or hasattr(res, "predict_proba") or callable(res):
                    return res
        # explicit callables if stashed
        for k in ("predict_proba_fn", "predict_fn"):
            if k in m and callable(m[k]):
                return m[k]
        # common generic keys
        common = [
            "best_estimator_", "best_estimator",
            "estimator_", "estimator",
            "model_", "model",
            "clf_", "clf",
            "final_model", "pipeline", "pipe",
            "sk_model", "sk_estimator", "base_estimator",
            "inner", "wrapped", "module", "net", "network",
        ]
        for k in common:
            if k in m:
                res = _extract_model(m[k], depth + 1, seen)
                if hasattr(res, "predict") or hasattr(res, "predict_proba") or callable(res):
                    return res
        # fallback: scan all values
        for v in m.values():
            res = _extract_model(v, depth + 1, seen)
            if hasattr(res, "predict") or hasattr(res, "predict_proba") or callable(res):
                return res
        return m

    # tuples/lists
    if isinstance(m, (list, tuple)):
        for v in m:
            res = _extract_model(v, depth + 1, seen)
            if hasattr(res, "predict") or hasattr(res, "predict_proba") or callable(res):
                return res
        return m

    # generic object: probe common attributes
    for attr in [
        "model", "model_", "clf", "clf_", "estimator", "estimator_",
        "pipeline", "pipe", "sk_model", "sk_estimator", "base_estimator",
        "final_model", "inner", "wrapped", "module", "net", "network",
    ]:
        if hasattr(m, attr):
            res = _extract_model(getattr(m, attr), depth + 1, seen)
            if hasattr(res, "predict") or hasattr(res, "predict_proba") or callable(res):
                return res

    return m

def _unwrap_model(m):
    em = _extract_model(m)
    is_pipe = (not callable(em)) and _is_pipeline(em)
    return em, is_pipe

def _predict_fn(model):
    em, _ = _unwrap_model(model)
    if callable(em):
        return em
    if hasattr(em, "predict_proba"):
        return lambda X: em.predict_proba(X)
    return lambda X: em.predict(X)

def _is_tree_model(model) -> bool:
    em, is_pipe = _unwrap_model(model)
    if is_pipe or callable(em):
        return False
    n = em.__class__.__name__.lower()
    return ("randomforest" in n or "xgb" in n or "xgboost" in n
            or "gradientboost" in n or "extratrees" in n or "catboost" in n)

def _is_linear(model) -> bool:
    em, is_pipe = _unwrap_model(model)
    return (not is_pipe) and (not callable(em)) and hasattr(em, "coef_")

def _choose_explainer(model, masker):
    em, is_pipe = _unwrap_model(model)
    if not callable(em) and _is_tree_model(em):
        return shap.TreeExplainer(em)
    if not callable(em) and _is_linear(em):
        return shap.LinearExplainer(em, masker)
    # pipelines, callables, calibrated wrappers, neural nets, tabnet/xlstm → generic
    return shap.Explainer(_predict_fn(em), masker)

def _to_df(X, feature_names):
    if isinstance(X, pd.DataFrame):
        return X[feature_names] if feature_names is not None else X
    return pd.DataFrame(X, columns=feature_names)

def _pick_class_axis(values: np.ndarray, n_features: int):
    """
    SHAP 3D values may be (n, classes, p) or (n, p, classes).
    Return tuple: (class_axis, feat_axis).
    """
    if values.shape[1] == n_features:
        return 2, 1  # (n, p, classes)
    if values.shape[2] == n_features:
        return 1, 2  # (n, classes, p)
    return 2, 1  # fallback

def _reduce_classes(values, model, X):
    """
    Reduce to (n, p). Handles (n, p), (n, classes, p), (n, p, classes).
    """
    if values is None:
        return values

    values = np.asarray(values)
    if values.ndim == 2:
        return values

    if values.ndim == 3:
        n_features = X.shape[1]
        class_axis, feat_axis = _pick_class_axis(values, n_features)

        em, _ = _unwrap_model(model)
        # Prob-weighted aggregation when possible
        if hasattr(em, "predict_proba"):
            proba = np.asarray(em.predict_proba(X))
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            n_classes = values.shape[class_axis]
            if proba.shape[1] != n_classes:
                return values.mean(axis=class_axis)  # fallback
            if class_axis == 1 and feat_axis == 2:      # (n, classes, p)
                return (proba[:, :, None] * values).sum(axis=1)
            else:                                       # (n, p, classes)
                return (proba[:, None, :] * values).sum(axis=2)

        # Else: use predicted class if available
        if hasattr(em, "classes_"):
            preds = em.predict(X)
            c2i = {c: i for i, c in enumerate(em.classes_)}
            idx = np.array([c2i.get(p, 0) for p in preds], dtype=int)
            if class_axis == 1 and feat_axis == 2:      # (n, classes, p)
                return values[np.arange(values.shape[0]), idx, :]
            else:                                       # (n, p, classes)
                return np.stack([values[i, :, idx[i]] for i in range(values.shape[0])], axis=0)

        return values.mean(axis=class_axis)  # conservative fallback

    return np.squeeze(values)

# ================= public API =================
def write_shap_reports(models: dict,
                       X_by_model: dict,
                       feature_names: list,
                       CONFIGS: dict,
                       verbose: bool = True) -> str:
    """
    Saves:
      - GLOBAL:  DIR_tr_va_te_metric_shap_SAVE_DIR/global_shap_<RUN_TS>.csv
      - LOCAL:   DIR_tr_va_te_metric_shap_SAVE_DIR/<model>/local_shap_<RUN_TS>.csv
    Also prints:
      "SHAP Analysis for <Pretty Name> (<key>) starting ..."
      "..."
      "SHAP Analysis for <Pretty Name> (<key>) end."
      "SHAP Analysis for <Pretty Name> (<key>) result saved on: <path> saved"
    and a final summary for all models.
    """
    if not models:
        raise ValueError("No models provided.")
    shap_base = _shap_dir(CONFIGS)
    ts_suffix = _run_ts(CONFIGS)

    # feature list
    if feature_names is None:
        X0 = X_by_model.get("default")
        if X0 is None:
            first_key = next(iter(models))
            X0 = X_by_model.get(first_key)
            if X0 is None:
                raise ValueError("feature_names is None and couldn't infer columns from X_by_model.")
        feature_names = list(getattr(X0, "columns", [f"f{i}" for i in range(np.asarray(X0).shape[1])]))

    global_df = pd.DataFrame({"feature": feature_names})
    processed_keys = []

    for key, raw_model in models.items():
        label = f"{_MODEL_LABELS.get(key, key.upper())} ({key})"
        try:
            if verbose:
                print(f'\n =====> SHAP Analysis (permutation) for {label} starting ... <=====')

            # choose test matrix
            X = X_by_model.get(key, X_by_model.get("default"))
            if X is None:
                raise ValueError(f"No X_test for '{key}' and no 'default' provided.")
            X = _to_df(X, feature_names)

            masker = _bg_masker(X, k=200, seed=0)
            explainer = _choose_explainer(raw_model, masker)

            em, _ = _unwrap_model(raw_model)
            is_generic = not (_is_tree_model(em) or _is_linear(em))
            if is_generic:
                sv = explainer(X, max_evals=min(1000, 10 * X.shape[1]))
            else:
                sv = explainer(X)

            values = getattr(sv, "values", sv)
            values = _reduce_classes(values, em, X)  # -> (n, p)

            # ----- LOCAL save (ensure per-model directory exists) -----
            mdir = _model_dir(shap_base, key)
            local_df = pd.DataFrame(values, columns=feature_names)
            local_fp = os.path.join(mdir, f"local_shap{ts_suffix}.csv")
            local_df.to_csv(local_fp, index=False)

            # ----- GLOBAL aggregates for this model -----
            mean_signed = local_df.mean(axis=0).to_numpy()
            mean_abs = local_df.abs().mean(axis=0).to_numpy()
            denom = (mean_abs.sum() + 1e-12)
            norm_abs = mean_abs / denom

            global_df[f"{key}_shap"] = mean_signed
            global_df[f"{key}_shap_norm"] = norm_abs
            global_df[f"{key}_shap_abs"] = mean_abs

            if verbose:
                print(f' =====> SHAP Analysis for {label} result saved on: {_normpath(local_fp)} saved <=====')
                print(f' =====> SHAP Analysis for {label} end. <=====')
            processed_keys.append(key)

        except Exception as e:
            # Keep going for other models; print a clear note for this one.
            print(f'##### [WARN] SHAP Analysis for {label} failed: {e} #####')

    # Save GLOBAL only if at least one model succeeded
    global_fp = os.path.join(shap_base, f"global_shap{ts_suffix}.csv")
    if processed_keys:
        global_df.to_csv(global_fp, index=False)
        if verbose:
            keys_str = ", ".join(processed_keys)
            print(f' =====> SHAP Analysis for all models ({keys_str}) done and result saved on: {_normpath(global_fp)} saved <=====')
    else:
        # nothing processed; still write header-only global
        global_df.to_csv(global_fp, index=False)
        if verbose:
            print(f'##### [WARN] No model SHAP computed. Wrote empty global scaffold to: {_normpath(global_fp)} #####')

    return global_fp
