# utils/_2_ml_lr.py

# (optional) shebang/coding cookie and/or module docstring can be here
# -*- coding: utf-8 -*-
"""Logistic Regression pipeline utils."""

from __future__ import annotations  # <-- must come first (after docstring)

import os
from typing import Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    brier_score_loss
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib

__all__ = ["train_logistic_regression", "run_lr_pipeline_compat", "run_logreg_pipeline"]
# ...rest of the file unchanged...



# --------------------------------------------
# 0) Small helpers
# --------------------------------------------
def _ensure_1d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.ravel()
    return a


def _pick_solver(penalty: str, l1_ratio: Optional[float]) -> str:
    """Choose a compatible solver given penalty/l1_ratio."""
    penalty = (penalty or "l2").lower()
    if penalty == "none":
        return "lbfgs"
    if penalty == "l1":
        return "liblinear"           # or 'saga' for large/sparse data
    if penalty == "l2":
        return "lbfgs"               # 'liblinear' also fine for small data
    if penalty == "elasticnet":
        return "saga"
    raise ValueError(f"Unsupported penalty: {penalty}")


# --------------------------------------------
# 1) Training
# --------------------------------------------
def train_logistic_regression(
    X_train, y_train,
    *,
    penalty: str = "l2",            # {'l1','l2','elasticnet','none'}
    C: float = 1.0,                 # inverse regularization strength
    l1_ratio: Optional[float] = None,  # for elasticnet [0,1]
    class_weight: Optional[str|dict] = "balanced",
    max_iter: int = 2000,
    n_jobs: int = -1,
    random_state: int = 42,
    scale: bool = True
) -> Pipeline:
    """
    Returns a fitted Pipeline([StandardScaler?, LogisticRegression]).
    """
    solver = _pick_solver(penalty, l1_ratio)
    lr = LogisticRegression(
        penalty=None if penalty == "none" else penalty,
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        n_jobs=n_jobs if solver in ("saga",) else None,  # lbfgs/liblinear ignore n_jobs
        random_state=random_state,
    )
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("clf", lr))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    return pipe


# --------------------------------------------
# 2) Predict
# --------------------------------------------
def predict_proba(model: Pipeline, X) -> np.ndarray:
    """Return positive-class probability (shape: [n_samples])."""
    p = model.predict_proba(X)[:, 1]
    return _ensure_1d(p)


def predict_label(model: Pipeline, X, threshold: float = 0.5) -> np.ndarray:
    """Threshold probabilities to labels."""
    p = predict_proba(model, X)
    return (p >= threshold).astype(int)


# --------------------------------------------
# 3) Evaluation (single threshold)
# --------------------------------------------
def evaluate_at_threshold(
    y_true, y_proba, threshold: float = 0.5
) -> Dict[str, float]:
    y_true = _ensure_1d(y_true)
    y_proba = _ensure_1d(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
    }
    return metrics


def classification_text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=3, zero_division=0)


# --------------------------------------------
# 4) Threshold sweep (with optional cost function)
# --------------------------------------------
def sweep_thresholds(
    y_true,
    y_proba,
    thresholds: Optional[Iterable[float]] = None,
    cost_fn=None
) -> pd.DataFrame:
    """
    Build a DataFrame of metrics across thresholds.
    cost_fn signature: cost = cost_fn(y_true, y_pred)
    """
    y_true = _ensure_1d(y_true)
    y_proba = _ensure_1d(y_proba)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    roc_auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        row = {
            "thr": thr,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "acc": accuracy_score(y_true, y_pred),
            "bacc": balanced_accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc,
            "avg_precision": ap,
        }
        if cost_fn is not None:
            row["cost"] = float(cost_fn(y_true, y_pred))
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["thr", "precision", "recall", "f1", "acc", "bacc", "roc_auc", "avg_precision"]
    if cost_fn is not None:
        cols.append("cost")
    return df[cols]


# --------------------------------------------
# 5) Curves (ROC / PR)
# --------------------------------------------
def plot_roc_pr(y_true, y_proba, title_suffix: str = "") -> None:
    y_true = _ensure_1d(y_true)
    y_proba = _ensure_1d(y_proba)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0,1], [0,1], ls="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve{(' - ' + title_suffix) if title_suffix else ''}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve{(' - ' + title_suffix) if title_suffix else ''}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# 6) Cross-Validation + Search
# --------------------------------------------
def cv_search_logreg(
    X, y,
    *,
    scale: bool = True,
    class_weight: Optional[str|dict] = "balanced",
    random_state: int = 42,
    n_splits: int = 5,
    n_jobs: int = -1
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Grid-search over common Logistic Regression settings.
    Refit on best 'roc_auc'. Returns (best_estimator_, cv_results_df).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()) if scale else ("passthrough", "passthrough"),
        ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=random_state))
    ])

    param_grid = [
        # L2 regularization
        {
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
            "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        # L1 regularization
        {
            "clf__penalty": ["l1"],
            "clf__solver": ["liblinear"],
            "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        # Elastic Net (requires saga)
        {
            "clf__penalty": ["elasticnet"],
            "clf__solver": ["saga"],
            "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "clf__C": [0.1, 1.0, 3.0],
        },
        # No penalty (rare; completeness)
        {
            "clf__penalty": ["none"],
            "clf__solver": ["lbfgs"]
        }
    ]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision"
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
        verbose=0
    )
    gs.fit(X, y)

    results = pd.DataFrame(gs.cv_results_).sort_values("mean_test_roc_auc", ascending=False)
    return gs.best_estimator_, results


# --------------------------------------------
# 7) Optional: Probability calibration
# --------------------------------------------
def calibrate_probabilities(
    model: Pipeline,
    X_train, y_train,
    method: str = "isotonic",  # 'sigmoid' or 'isotonic'
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Wrap an existing (already-fitted) pipeline in a calibrated model.
    """
    base = model
    calibrated = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


# --------------------------------------------
# 8) Persistence
# --------------------------------------------
def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


# --------------------------------------------
# 9) Orchestrator (simple)
# --------------------------------------------
def run_logreg_pipeline(
    X_train, y_train, X_val, y_val, X_test, y_test,
    *,
    penalty: str = "l2",
    C: float = 1.0,
    l1_ratio: Optional[float] = None,
    class_weight: Optional[str|dict] = "balanced",
    scale: bool = True,
    random_state: int = 42
) -> Dict[str, object]:
    """
    Trains LR, evaluates on val/test, returns dict of artifacts.
    """
    model = train_logistic_regression(
        X_train, y_train,
        penalty=penalty, C=C, l1_ratio=l1_ratio,
        class_weight=class_weight, scale=scale, random_state=random_state
    )

    # Probabilities
    p_val = predict_proba(model, X_val)
    p_test = predict_proba(model, X_test)

    # Default threshold 0.5 eval
    m_val = evaluate_at_threshold(y_val, p_val, threshold=0.5)
    m_test = evaluate_at_threshold(y_test, p_test, threshold=0.5)

    # Threshold sweep (no custom cost by default)
    sweep_val = sweep_thresholds(y_val, p_val)
    sweep_test = sweep_thresholds(y_test, p_test)

    # Reports at 0.5
    y_val_pred = (p_val >= 0.5).astype(int)
    y_test_pred = (p_test >= 0.5).astype(int)
    rpt_val = classification_text_report(y_val, y_val_pred)
    rpt_test = classification_text_report(y_test, y_test_pred)

    artifacts = {
        "model": model,
        "val": {"proba": p_val, "metrics": m_val, "sweep": sweep_val, "report": rpt_val},
        "test": {"proba": p_test, "metrics": m_test, "sweep": sweep_test, "report": rpt_test},
    }
    return artifacts


# --------------------------------------------
# 10) RF-compatible wrapper (same calling style)
# --------------------------------------------
def _pick_threshold_from_sweep(sweep_df: pd.DataFrame, strategy: str = "f1") -> float:
    strategy = (strategy or "f1").lower()
    if strategy in sweep_df.columns:
        i = sweep_df[strategy].to_numpy().argmax()
        return float(sweep_df.iloc[i]["thr"])
    return 0.5


def _pack_results(y_true, proba, sweep_df, chosen_thr):
    y_true = _ensure_1d(y_true)
    proba = _ensure_1d(proba)
    y_pred = (proba >= chosen_thr).astype(int)
    return {
        "threshold": chosen_thr,
        "proba": proba,
        "pred": y_pred,
        "sweep": sweep_df,
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
            "avg_precision": average_precision_score(y_true, proba),
        },
        "report": classification_report(y_true, y_pred, digits=3, zero_division=0),
    }


def run_lr_pipeline_compat(
    X_train, y_train, X_val, y_val, X_test, y_test,
    features=None,  # kept for signature parity; unused here
    *,
    class_weight="balanced",
    penalty="l2",
    C=1.0,
    l1_ratio=None,
    scale=True,
    threshold_strategy="f1",
    cost_fn=None,   # optional: if you have one, sweep will include 'cost'
    random_state=42
):
    """
    Mirrors your RF call style:
    model_lr, results_lr = run_lr_pipeline_compat(..., threshold_strategy='f1')
    """
    model = train_logistic_regression(
        X_train, y_train,
        penalty=penalty, C=C, l1_ratio=l1_ratio,
        class_weight=class_weight, scale=scale, random_state=random_state
    )

    # probs
    p_val  = predict_proba(model, X_val)
    p_test = predict_proba(model, X_test)

    # sweeps
    sweep_val  = sweep_thresholds(y_val,  p_val,  cost_fn=cost_fn)
    sweep_test = sweep_thresholds(y_test, p_test, cost_fn=cost_fn)

    # choose threshold on VAL, lock for TEST
    chosen_strategy = ("cost" if (cost_fn is not None and "cost" in sweep_val.columns and threshold_strategy == "cost")
                       else threshold_strategy)
    chosen_thr = _pick_threshold_from_sweep(sweep_val, strategy=chosen_strategy)

    # pack outputs
    results = {
        "val":  _pack_results(y_val,  p_val,  sweep_val,  chosen_thr),
        "test": _pack_results(y_test, p_test, sweep_test, chosen_thr),
        "threshold_strategy": threshold_strategy,
    }
    return model, results


# --------------------------------------------
# 11) Tiny export helpers (optional)
# --------------------------------------------
from datetime import datetime
import os
import pandas as pd
from typing import Dict, Optional

def print_lr_metrics(results: Dict[str, dict], digits: int = 4) -> pd.DataFrame:
    """
    Pretty-print VAL/TEST metrics from results and return them as a DataFrame.
    """
    val = results["val"]["metrics"]
    test = results["test"]["metrics"]

    # Choose a stable column order (add/remove as you like)
    cols = [
        "accuracy", "balanced_accuracy", "f1",
        "precision", "recall", "roc_auc", "avg_precision"
    ]
    df = pd.DataFrame([val, test], index=["val", "test"])[cols]
    df_rounded = df.round(digits)

    # Console print
    print("\n=== Logistic Regression Metrics ===")
    print(df_rounded.to_string())
    print(f"\nChosen threshold (from VAL): {results['val']['threshold']:.4f}")
    return df

def export_lr_artifacts(
    results: Dict[str, dict],
    outdir: str,
    prefix: str = "lr",
    timestamp: Optional[str] = None,
    include_preds: bool = False,
    print_metrics: bool = True,            # <--- new
    combined_metrics_filename: Optional[str] = None  # override name if you want
) -> Dict[str, str]:
    """
    Saves LR sweeps, metrics, and reports to CSV/TXT in `outdir`,
    with a timestamp appended to each filename. Optionally prints metrics.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    # 1) (optional) print metrics to console
    if print_metrics:
        _ = print_lr_metrics(results)

    # 2) sweeps
    p = os.path.join(outdir, f"{prefix}_val_sweep_{ts}.csv")
    results["val"]["sweep"].to_csv(p, index=False); paths["val_sweep"] = p
    p = os.path.join(outdir, f"{prefix}_test_sweep_{ts}.csv")
    results["test"]["sweep"].to_csv(p, index=False); paths["test_sweep"] = p

    # 3) metrics (single rows)
    p = os.path.join(outdir, f"{prefix}_val_metrics_{ts}.csv")
    pd.DataFrame([results["val"]["metrics"]]).to_csv(p, index=False); paths["val_metrics"] = p
    p = os.path.join(outdir, f"{prefix}_test_metrics_{ts}.csv")
    pd.DataFrame([results["test"]["metrics"]]).to_csv(p, index=False); paths["test_metrics"] = p

    # 3b) combined metrics table (val+test in one file)
    combined = pd.DataFrame(
        [results["val"]["metrics"], results["test"]["metrics"]],
        index=["val", "test"]
    )
    comb_name = combined_metrics_filename or f"{prefix}_combined_metrics_{ts}.csv"
    p = os.path.join(outdir, comb_name)
    combined.to_csv(p)
    paths["combined_metrics"] = p

    # 4) reports (txt)
    p = os.path.join(outdir, f"{prefix}_val_report_{ts}.txt")
    with open(p, "w") as f: f.write(results["val"]["report"]); paths["val_report"] = p
    p = os.path.join(outdir, f"{prefix}_test_report_{ts}.txt")
    with open(p, "w") as f: f.write(results["test"]["report"]); paths["test_report"] = p

    # 5) (optional) predictions/probabilities
    if include_preds:
        p = os.path.join(outdir, f"{prefix}_val_proba_pred_{ts}.csv")
        pd.DataFrame({"proba": results["val"]["proba"], "pred": results["val"]["pred"]}).to_csv(p, index=False)
        paths["val_proba_pred"] = p

        p = os.path.join(outdir, f"{prefix}_test_proba_pred_{ts}.csv")
        pd.DataFrame({"proba": results["test"]["proba"], "pred": results["test"]["pred"]}).to_csv(p, index=False)
        paths["test_proba_pred"] = p

    return paths

# --------------------------------------------
# 12) Minimal usage example (comment out in library)
# --------------------------------------------
# if __name__ == "__main__":
    # Example (pseudo):
    # X_train, X_val, X_test = ...
    # y_train, y_val, y_test = ...

    # model, results = run_lr_pipeline_compat(
    #     X_train, y_train, X_val, y_val, X_test, y_test,
    #     features=None,
    #     class_weight="balanced",
    #     penalty="l2",
    #     C=1.0,
    #     threshold_strategy="f1",
    # )
    # print(results["test"]["metrics"])
    # export_lr_artifacts(results, outdir="op/3_models/lr_run_001")
    # pass
