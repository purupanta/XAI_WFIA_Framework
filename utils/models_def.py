# utils/models_def.py
from __future__ import annotations

# ===== Public API =====
__all__ = [
    # trainers
    "train_models", "train_all_models",
    "fit_lr", "fit_rf", "fit_xgb", "fit_mlp", "fit_tabnet",
    # preds / thresholds
    "predict_proba_safe", "optimize_threshold",
    # eval
    "evaluate_split", "evaluate_all", "evaluate_all_splits",
    "summarize_table", "summarize_metrics_table",
    # plotting
    "plot_metric_curves", "plot_train_val_loss", "plot_roc_all", "plot_pr_all",
    # optional compatibility
    "calibrate_model_isotonic", "AveragingEnsemble",
]

# ===== Imports =====
import warnings
from typing import Dict, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    log_loss, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

def stamp(msg: str) -> None:
    print(msg)

# ===== Small utilities =====
def _class_ratio(y) -> tuple[int, int, float]:
    y = np.asarray(y).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    r = (pos / max(neg, 1.0)) if neg else 0.0
    return pos, neg, r

def _balanced_sample_weight(y):
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum(); neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)
    w_pos = 0.5 / pos; w_neg = 0.5 / neg
    sw = np.where(y == 1, w_pos, w_neg)
    return (sw * y.size).astype(np.float32)

def _scale_pos_weight(y):
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum(); neg = (y == 0).sum()
    return (neg / max(pos, 1)) if pos > 0 else 1.0

def predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 1: p = np.column_stack([1 - p, p])
        elif p.shape[1] == 1: p = np.column_stack([1 - p[:, 0], p[:, 0]])
        return p
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X)).reshape(-1)
        prob1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - prob1, prob1])
    raise AttributeError("Model does not support probability predictions.")

def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

# ====== Fitters (short, explicit, robust) ======
def fit_lr(X_tr, y_tr, *, max_iter=400, C=1.0, penalty="l2", random_state=42):
    """Logistic Regression in a pipeline with StandardScaler (prevents scale mismatch)."""
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    l1_ratio = 0.15 if penalty == "elasticnet" else None
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=float(C), penalty=penalty, solver=solver,
            l1_ratio=l1_ratio, class_weight="balanced",
            max_iter=int(max_iter), random_state=int(random_state),
            n_jobs=-1 if solver == "saga" else None
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X_tr, y_tr)
    stamp("[LR] trained (pipeline with StandardScaler).")
    return pipe

def fit_rf(X_tr, y_tr, *, n_estimators=600, max_depth=None, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=max_depth, max_features="sqrt",
        min_samples_leaf=1, class_weight="balanced_subsample",
        n_jobs=-1, random_state=int(random_state)
    )
    rf.fit(X_tr, y_tr)
    stamp("[RF] trained.")
    return rf

def fit_xgb(X_tr, y_tr, X_val=None, y_val=None, *,
            n_estimators=1200, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3.0,
            random_state=42):
    if XGBClassifier is None:
        stamp("[XGB] not installed → skipping.")
        return None, {}
    spw = _scale_pos_weight(y_tr)
    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=float(min_child_weight),
        reg_lambda=1.0,
        tree_method="hist", predictor="auto",
        n_jobs=-1, random_state=int(random_state),
        scale_pos_weight=spw,
        eval_metric=["auc", "logloss"]
    )
    eval_set = [(X_tr, y_tr)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    xgb.fit(
        X_tr, y_tr,
        eval_set=eval_set,
        verbose=False,
        early_stopping_rounds=200 if len(eval_set) > 1 else None
    )
    hist = {}
    try:
        er = xgb.evals_result()
        for split, metrics in er.items():
            for k, v in metrics.items():
                hist[f"{split}_{k}"] = list(v)
        # pick max length across any metric for x-axis
        max_len = 0
        for m in er.values():
            for s in m.values():
                max_len = max(max_len, len(s))
        hist["x"] = list(range(1, max_len + 1))
    except Exception:
        pass
    stamp("[XGB] trained.")
    return xgb, hist

def fit_mlp(X_tr, y_tr, *, hidden=(128, 64), max_iter=300, random_state=42):
    """MLP in a pipeline with StandardScaler. Early stopping enabled."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(hidden), activation="relu", solver="adam",
            alpha=2e-4, learning_rate_init=8e-4, max_iter=int(max_iter),
            early_stopping=True, n_iter_no_change=20, random_state=int(random_state)
        )),
    ])
    sw = _balanced_sample_weight(y_tr)
    try:
        pipe.fit(X_tr, y_tr, mlp__sample_weight=sw)  # some sklearns accept this
        stamp("[MLP] trained (with sample_weight).")
    except Exception:
        pipe.fit(X_tr, y_tr)
        stamp("[MLP] trained (no sample_weight supported).")
    return pipe

def fit_tabnet(X_tr, y_tr, X_val, y_val, *,
               width=24, n_steps=4, max_epochs=120, patience=40,
               batch_size=256, virtual_batch_size=32,
               random_state=42, use_gpu=True):
    """TabNet (numpy float32). Works even if pytorch-tabnet missing (returns None, {})."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except Exception:
        stamp("[TabNet] pytorch-tabnet not installed → skipping.")
        return None, {}

    def _f32(a):
        a = a.values if hasattr(a, "values") else a
        a = np.asarray(a, dtype=np.float32)
        a[~np.isfinite(a)] = 0.0
        return a

    Xtr = _f32(X_tr); Xva = _f32(X_val)
    ytr = np.asarray(y_tr).astype(int); yva = np.asarray(y_val).astype(int)

    device = "cuda" if (use_gpu and _torch_cuda()) else "cpu"
    tn = TabNetClassifier(
        n_d=width, n_a=width, n_steps=n_steps, gamma=1.5,
        n_shared=2, n_independent=2, lambda_sparse=1e-5,
        verbose=1, seed=int(random_state), device_name=device
    )
    stamp(f"[TabNet] training on {device}...")
    tn.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)], eval_name=["val"], eval_metric=["auc", "logloss"],
        patience=int(patience), max_epochs=int(max_epochs),
        batch_size=int(batch_size), virtual_batch_size=int(virtual_batch_size),
        num_workers=0
    )
    hist = {}
    try:
        if hasattr(tn, "history") and isinstance(tn.history, dict):
            H = tn.history
            if "loss" in H: hist["loss"] = list(H["loss"])
            for k in ("val_0_auc", "val_0_logloss", "val_auc", "val_logloss"):
                if k in H: hist[k] = list(H[k])
            max_len = max((len(v) for v in hist.values()), default=0)
            hist["x"] = list(range(1, max_len + 1))
    except Exception:
        pass
    stamp("[TabNet] trained.")
    return tn, hist

# ====== Optional: calibration & simple average ensemble (kept for compatibility) ======
def calibrate_model_isotonic(model, X_val, y_val):
    cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    cal.fit(X_val, y_val)
    return cal

class AveragingEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] | None = None):
        self.models = models
        if weights is None:
            keys = list(models.keys())
            self.weights = {k: 1.0/len(keys) for k in keys}
        else:
            self.weights = weights
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def predict_proba(self, X):
        p = None
        for name, mdl in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w <= 0: continue
            pi = predict_proba_safe(mdl, X)[:, 1]
            p = (w * pi) if p is None else (p + w * pi)
        if p is None:
            raise RuntimeError("Ensemble has no positive weights.")
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

# ====== Evaluation (simple & explicit) ======
def evaluate_split(models: Dict[str, Any], X, y, *, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    if X is None or y is None:
        return {}
    res: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        if m is None:
            continue
        try:
            proba = predict_proba_safe(m, X)[:, 1]
            pred = (proba >= float(threshold)).astype(int)
            res[name] = {
                "auc": float(roc_auc_score(y, proba)),
                "ap": float(average_precision_score(y, proba)),
                "logloss": float(log_loss(y, proba, labels=[0, 1])),
                "acc": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred)),
            }
        except Exception as e:
            stamp(f"[EVAL/{name}] skipped: {e}")
    return res

def evaluate_all(models: Dict[str, Any],
                 *, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "train": evaluate_split(models, X_train, y_train),
        "val":   evaluate_split(models, X_val, y_val) if (X_val is not None and y_val is not None) else {},
        "test":  evaluate_split(models, X_test, y_test) if (X_test is not None and y_test is not None) else {},
    }

# Backwards-compatible alias
def evaluate_all_splits(models: Dict[str, Any], *, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
                        per_model_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    res = evaluate_all(models, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    # If thresholds provided, add acc@t / f1@t
    if per_model_thresholds:
        for split_name, dct in res.items():
            X_split = {"train": X_train, "val": X_val, "test": X_test}[split_name]
            y_split = {"train": y_train, "val": y_val, "test": y_test}[split_name]
            if X_split is None or y_split is None:
                continue
            for name in list(dct.keys()):
                try:
                    t = float(per_model_thresholds.get(name, 0.5))
                    proba = predict_proba_safe(models[name], X_split)[:, 1]
                    pred = (proba >= t).astype(int)
                    dct[name][f"acc@{t:.2f}"] = float(accuracy_score(y_split, pred))
                    dct[name][f"f1@{t:.2f}"]  = float(f1_score(y_split, pred))
                except Exception:
                    pass
    return res

def summarize_table(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    names = sorted(set(
        list(all_metrics.get("train", {}).keys())
        + list(all_metrics.get("val", {}).keys())
        + list(all_metrics.get("test", {}).keys())
    ))
    lines: list[str] = []
    for split in ("train", "val", "test"):
        if not all_metrics.get(split):
            continue
        lines.append(f"\n=== {split.upper()} ===")
        lines.append("model       auc     ap      logloss   acc     f1")
        for m in names:
            met = all_metrics[split].get(m)
            if not met:
                continue
            lines.append(
                f"{m:<10s} "
                f"{met.get('auc', float('nan')):<7.3f} "
                f"{met.get('ap', float('nan')):<7.3f} "
                f"{met.get('logloss', float('nan')):<8.3f} "
                f"{met.get('acc', float('nan')):<7.3f} "
                f"{met.get('f1', float('nan')):<7.3f}"
            )
    return "\n".join(lines)

# Backwards-compatible alias
def summarize_metrics_table(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    return summarize_table(all_metrics)

def optimize_threshold(model, X_val, y_val, *, metric="f1") -> float:
    p = predict_proba_safe(model, X_val)[:, 1]
    grid = np.linspace(0.05, 0.95, 19)
    if str(metric).lower() == "youden":
        from sklearn.metrics import recall_score
        def scorer(t):
            yhat = (p >= t).astype(int)
            tn = np.sum((y_val == 0) & (yhat == 0))
            fp = np.sum((y_val == 0) & (yhat == 1))
            spec = tn / (tn + fp + 1e-9)
            sens = recall_score(y_val, yhat)
            return sens + spec - 1.0
    else:
        def scorer(t):
            return f1_score(y_val, (p >= t).astype(int))
    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = scorer(float(t))
        if s > best_s:
            best_s, best_t = s, float(t)
    stamp(f"[THRESH] best_t={best_t:.3f}, score={best_s:.4f}")
    return float(best_t)

# ====== Plots ======
def plot_roc_all(models: Dict[str, Any], X, y, split_name="Validation"):
    plt.figure()
    for name, m in models.items():
        if m is None:
            continue
        try:
            p = predict_proba_safe(m, X)[:, 1]
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.title(f"ROC — {split_name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.grid(True); plt.legend(); plt.tight_layout()

def plot_pr_all(models: Dict[str, Any], X, y, split_name="Validation"):
    plt.figure()
    for name, m in models.items():
        if m is None:
            continue
        try:
            p = predict_proba_safe(m, X)[:, 1]
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    plt.title(f"Precision–Recall — {split_name}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True); plt.legend(); plt.tight_layout()

def plot_metric_curves(histories: Dict[str, Dict[str, list]]):
    """
    Plots training/validation curves for XGB, TabNet, and MLP (if curves exist).
    Expects histories keyed by model name (xgb/tabnet/mlp) with dicts of series.
    """
    for name, hist in histories.items():
        if not hist:
            continue
        x = hist.get("x", list(range(1, max((len(v) for v in hist.values()), default=0) + 1)))
        # draw any metric series available
        drew = False
        plt.figure()
        for key, series in hist.items():
            if key == "x":
                continue
            try:
                plt.plot(x[:len(series)], series, label=f"{name}:{key}")
                drew = True
            except Exception:
                pass
        if not drew:
            plt.close()
            continue
        plt.title(f"{name} — training/validation metrics")
        plt.xlabel("Iteration / Epoch")
        plt.grid(True); plt.legend(); plt.tight_layout()

def plot_train_val_loss(histories: Dict[str, Dict[str, list]]):
    """
    Focused plot on loss/logloss where present.
    """
    for name, hist in histories.items():
        if not hist:
            continue
        x = hist.get("x", list(range(1, max((len(v) for v in hist.values()), default=0) + 1)))
        keys = [k for k in hist.keys() if ("loss" in k and k != "x")]
        if not keys:
            continue
        plt.figure()
        for k in keys:
            v = hist[k]
            plt.plot(x[:len(v)], v, label=f"{name}:{k}")
        plt.title(f"{name} — loss/logloss")
        plt.xlabel("Iteration / Epoch")
        plt.grid(True); plt.legend(); plt.tight_layout()

# ====== Orchestrators ======
def train_all_models(
    X_train_res, y_train_res,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    *, random_state=42
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, list]]]:
    """Trains LR, RF, XGB, MLP, TabNet (if available). Returns (models, histories)."""
    p, n, r = _class_ratio(y_train_res)
    stamp(f"[DATA] train_res: N={p+n}, pos={p}, neg={n}, pos/neg={r:.3f}")
    if y_val is not None:
        p2, n2, r2 = _class_ratio(y_val); stamp(f"[DATA] val: N={p2+n2}, pos/neg={r2:.3f}")
    if y_test is not None:
        p3, n3, r3 = _class_ratio(y_test); stamp(f"[DATA] test: N={p3+n3}, pos/neg={r3:.3f}")

    models: Dict[str, Any] = {}
    histories: Dict[str, Dict[str, list]] = {}

    # LR (with scaler)
    models["lr"] = fit_lr(X_train_res, y_train_res, random_state=random_state)

    # RF
    models["rf"] = fit_rf(X_train_res, y_train_res, random_state=random_state)

    # XGB
    xgb, xgb_hist = fit_xgb(X_train_res, y_train_res, X_val, y_val, random_state=random_state)
    models["xgb"] = xgb; histories["xgb"] = xgb_hist

    # MLP (with scaler)
    models["mlp"] = fit_mlp(X_train_res, y_train_res, random_state=random_state)
    try:
        mlp_loss = models["mlp"].named_steps["mlp"].loss_curve_
        histories["mlp"] = {"train_loss": list(mlp_loss), "x": list(range(1, len(mlp_loss)+1))}
    except Exception:
        histories["mlp"] = {}

    # TabNet (needs val for early stopping)
    if X_val is not None and y_val is not None:
        tn, tn_hist = fit_tabnet(X_train_res, y_train_res, X_val, y_val, random_state=random_state)
        models["tabnet"] = tn; histories["tabnet"] = tn_hist
    else:
        models["tabnet"] = None; histories["tabnet"] = {}

    return models, histories

# Backwards-compatible name
def train_models(CONFIGS: Dict[str, Any],
                 X_train_res_scaled, y_train_res,
                 X_train_res, X_val, y_val,
                 X_test=None, y_test=None,
                 models: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compatibility wrapper: ignores *_scaled and delegates to train_all_models with raw features."""
    rs = int(CONFIGS.get("RANDOM_STATE", 42)) if isinstance(CONFIGS, dict) else 42
    return train_all_models(
        X_train_res, y_train_res,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        random_state=rs
    )
