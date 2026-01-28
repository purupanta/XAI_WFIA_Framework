"""
Classic ML + TabNet pipeline for binary classification.

Models:
- Logistic Regression (lr)
- Random Forest (rf)
- XGBoost (xgb)           [optional, requires xgboost]
- Multi-layer Perceptron (mlp, sklearn)
- TabNet (tabnet)         [optional, requires pytorch-tabnet]

Inputs (you already have them):
    X_train_res_scaled, X_val_scaled, X_test_scaled
    X_train_res,        X_val,        X_test
    y_train_res,        y_val,        y_test

Conventions:
- Scaled sets used for: lr, mlp
- Unscaled sets used for: rf, xgb, tabnet

What it does per model:
- Fits the model (with imbalance-aware settings where applicable)
- Shows training curves:
    * XGB: train vs val logloss and (if available) AUC in the SAME plot per metric
    * MLP: train loss curve
    * TabNet: train vs val loss (and AUC if present) in the SAME plot per metric
- Plots Validation ROC (with ROC AUC)
- Plots Validation PR curve (with PR AUC)
- Picks best F1 threshold on Validation
- Prints Validation & Test classification reports using that threshold
- (Optional) Saves artifacts (plots, metrics JSON, confusion matrices, reports)

Example:
    from classic_ml_deep_pipeline import run_all

    results = run_all(
        X_train_res_scaled=X_train_res_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        X_train_res=X_train_res,
        X_val=X_val,
        X_test=X_test,
        y_train_res=y_train_res,
        y_val=y_val,
        y_test=y_test,
        models=("lr","rf","xgb","mlp","tabnet"),
        random_state=42,
        save_dir="tr_te_va_results",   # optional
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- sklearn models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

# Optional deps
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    _HAVE_TABNET = True
except Exception:
    _HAVE_TABNET = False


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def _to_numpy_xy(X, y) -> Tuple[np.ndarray, np.ndarray]:
    Xn = X.values if hasattr(X, "values") else np.asarray(X)
    yn = np.asarray(y).reshape(-1)
    return Xn, yn


def best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Grid-search threshold in [0,1] to maximize F1."""
    eps = 1e-12
    y_true = y_true.astype(int)
    thr_grid = np.linspace(0.0, 1.0, 1001)
    best = {"thr": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for t in thr_grid:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        if f1 > best["f1"]:
            best = {"thr": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return best["thr"], {"precision": best["precision"], "recall": best["recall"], "f1": best["f1"]}


def _print_header(title: str):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def _plot_single_curve(x, y, xlabel: str, ylabel: str, title: str, save_path: Optional[Path] = None):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def _plot_dual_curve(x1, y1, label1: str, x2, y2, label2: str,
                     xlabel: str, ylabel: str, title: str, save_path: Optional[Path] = None):
    plt.figure()
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str, save_path: Optional[Path] = None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    _plot_single_curve(fpr, tpr, "False Positive Rate", "True Positive Rate", f"{title} | ROC AUC={auc:.4f}", save_path)
    return float(auc)


def _report_block(name: str, y_true: np.ndarray, y_scores: np.ndarray, thr: float):
    y_pred = (y_scores >= thr).astype(int)
    acc = (y_pred == y_true).mean()
    print(f"[predict] N={len(y_true)} | threshold={thr:.2f} | Accuracy={acc:.4f}")
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=2))


# ------------------------------------------------------------
# Model builders
# ------------------------------------------------------------

def build_model(model_key: str, random_state: int = 42, *, scale_pos_weight: Optional[float] = None):
    model_key = model_key.lower()
    if model_key == "lr":
        return LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
            verbose=False,
        )
    if model_key == "xgb":
        if not _HAVE_XGB:
            raise ImportError("xgboost is not installed: pip install xgboost>=1.7 (or conda install -c conda-forge xgboost)")
        spw = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
        return XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=random_state,
            tree_method="hist",
            eval_metric=["logloss", "auc"],
            scale_pos_weight=spw,
            n_jobs=-1,
        )
    if model_key == "tabnet":
        if not _HAVE_TABNET:
            raise ImportError("pytorch-tabnet is not installed: pip install pytorch-tabnet")
        return TabNetClassifier(
            n_d=16, n_a=16, n_steps=4, gamma=1.2, lambda_sparse=1e-3,
            optimizer_params=dict(lr=2e-3), seed=random_state, verbose=0, device_name="auto"
        )
    raise ValueError(f"Unknown model_key: {model_key}")


# ------------------------------------------------------------
# Fit + evaluate per model
# ------------------------------------------------------------

@dataclass
class ModelResult:
    model: Any
    val_auc: float
    val_pr_auc: float
    best_thr: float
    best_thr_metrics: Dict[str, float]
    val_proba: np.ndarray
    test_proba: np.ndarray
    extras: Dict[str, Any]


def fit_and_eval(
    model_key: str,
    Xtr_scl: pd.DataFrame,
    Xva_scl: pd.DataFrame,
    Xte_scl: pd.DataFrame,
    Xtr_raw: pd.DataFrame,
    Xva_raw: pd.DataFrame,
    Xte_raw: pd.DataFrame,
    ytr: pd.Series,
    yva: pd.Series,
    yte: pd.Series,
    random_state: int = 42,
    save_dir: Optional[Path] = None,
) -> ModelResult:
    _print_header(f"Training: {model_key}")
    use_scaled = model_key in {"lr", "mlp"}
    Xtr = Xtr_scl if use_scaled else Xtr_raw
    Xva = Xva_scl if use_scaled else Xva_raw
    Xte = Xte_scl if use_scaled else Xte_raw

    Xtr_np, ytr_np = _to_numpy_xy(Xtr, ytr)
    Xva_np, yva_np = _to_numpy_xy(Xva, yva)
    Xte_np, yte_np = _to_numpy_xy(Xte, yte)

    # imbalance ratio for models that support it
    pos = float((ytr_np == 1).sum())
    neg = float((ytr_np == 0).sum())
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    model = build_model(model_key, random_state=random_state, scale_pos_weight=scale_pos_weight)

    plots_dir = None
    if save_dir is not None:
        plots_dir = (save_dir / model_key / "plots")

    # --- Fit + record training history where available ---
    if model_key == "xgb":
        eval_set = [(Xtr_np, ytr_np), (Xva_np, yva_np)]
        model.fit(Xtr_np, ytr_np, eval_set=eval_set, verbose=False)
        try:
            hist = model.evals_result()
            if plots_dir is not None:
                # Train vs Val Logloss
                tr_ll = hist.get("validation_0", {}).get("logloss")
                va_ll = hist.get("validation_1", {}).get("logloss")
                if tr_ll is not None and va_ll is not None:
                    it_tr = np.arange(len(tr_ll))
                    it_va = np.arange(len(va_ll))
                    _plot_dual_curve(
                        it_tr, tr_ll, "train",
                        it_va, va_ll, "valid",
                        "Iteration", "Logloss", "XGBoost Logloss (train vs val)",
                        plots_dir / "xgb_logloss_train_vs_val.png",
                    )
                # Train vs Val AUC
                tr_auc = hist.get("validation_0", {}).get("auc")
                va_auc = hist.get("validation_1", {}).get("auc")
                if tr_auc is not None and va_auc is not None:
                    it_tr = np.arange(len(tr_auc))
                    it_va = np.arange(len(va_auc))
                    _plot_dual_curve(
                        it_tr, tr_auc, "train",
                        it_va, va_auc, "valid",
                        "Iteration", "AUC", "XGBoost AUC (train vs val)",
                        plots_dir / "xgb_auc_train_vs_val.png",
                    )
        except Exception:
            pass

    elif model_key == "mlp":
        model.fit(Xtr_np, ytr_np)
        if hasattr(model, "loss_curve_"):
            _plot_single_curve(
                np.arange(len(model.loss_curve_)),
                model.loss_curve_,
                "Epoch",
                "Loss",
                "MLP Training Loss",
                plots_dir / "mlp_train_loss.png" if plots_dir is not None else None,
            )

    elif model_key == "tabnet":
        try:
            max_epochs = 200
            model.fit(
                Xtr_np, ytr_np,
                eval_set=[(Xtr_np, ytr_np), (Xva_np, yva_np)],
                eval_name=["train", "valid"],
                eval_metric=["auc"],
                max_epochs=max_epochs,
                patience=20,
                batch_size=512,
                virtual_batch_size=64,
            )
            try:
                hist = model.history
                tr_loss = hist.get("train_loss") or hist.get("loss")
                va_loss = hist.get("valid_loss") or hist.get("val_loss")
                if tr_loss is not None and va_loss is not None:
                    it_tr = np.arange(len(tr_loss))
                    it_va = np.arange(len(va_loss))
                    _plot_dual_curve(
                        it_tr, tr_loss, "train",
                        it_va, va_loss, "valid",
                        "Epoch", "Loss", "TabNet Loss (train vs val)",
                        plots_dir / "tabnet_loss_train_vs_val.png" if plots_dir is not None else None,
                    )
                tr_auc = hist.get("train_auc")
                va_auc = hist.get("valid_auc")
                if tr_auc is not None and va_auc is not None:
                    it_tr = np.arange(len(tr_auc))
                    it_va = np.arange(len(va_auc))
                    _plot_dual_curve(
                        it_tr, tr_auc, "train",
                        it_va, va_auc, "valid",
                        "Epoch", "AUC", "TabNet AUC (train vs val)",
                        plots_dir / "tabnet_auc_train_vs_val.png" if plots_dir is not None else None,
                    )
            except Exception:
                pass
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[TabNet] CUDA OOM — retrying on CPU with smaller batches...")
                cpu_model = TabNetClassifier(
                    n_d=16, n_a=16, n_steps=4, gamma=1.2, lambda_sparse=1e-3,
                    optimizer_params=dict(lr=2e-3), seed=random_state, verbose=0, device_name="cpu"
                )
                cpu_model.fit(
                    Xtr_np, ytr_np,
                    eval_set=[(Xtr_np, ytr_np), (Xva_np, yva_np)],
                    eval_name=["train", "valid"],
                    eval_metric=["auc"],
                    max_epochs=150, patience=15, batch_size=256, virtual_batch_size=32,
                )
                model = cpu_model
                try:
                    hist = model.history
                    tr_loss = hist.get("train_loss") or hist.get("loss")
                    va_loss = hist.get("valid_loss") or hist.get("val_loss")
                    if tr_loss is not None and va_loss is not None:
                        it_tr = np.arange(len(tr_loss))
                        it_va = np.arange(len(va_loss))
                        _plot_dual_curve(
                            it_tr, tr_loss, "train",
                            it_va, va_loss, "valid",
                            "Epoch", "Loss", "TabNet Loss (train vs val)",
                            plots_dir / "tabnet_loss_train_vs_val.png" if plots_dir is not None else None,
                        )
                except Exception:
                    pass
            else:
                raise
    else:
        # lr, rf
        model.fit(Xtr_np, ytr_np)
        print("(No epoch-wise history available for this model.)")

    # --- Predict probabilities ---
    val_proba = model.predict_proba(Xva_np)[:, 1]
    test_proba = model.predict_proba(Xte_np)[:, 1]

    # --- ROC + AUC (Validation) ---
    roc_path = plots_dir / f"{model_key}_val_roc.png" if plots_dir is not None else None
    val_auc = plot_roc_curve(yva_np, val_proba, title=f"{model_key.upper()} Validation ROC", save_path=roc_path)

    # --- PR curve + AUC (Validation) ---
    re, pr, _ = precision_recall_curve(yva_np, val_proba)   # x-axis recall, y-axis precision
    pr_auc = average_precision_score(yva_np, val_proba)
    _plot_single_curve(re, pr, "Recall", "Precision",
                       f"{model_key.upper()} Validation PR AUC={pr_auc:.4f}",
                       plots_dir / f"{model_key}_val_pr.png" if plots_dir is not None else None)

    # --- Choose best threshold on Validation by F1 ---
    thr, thr_metrics = best_threshold_f1(yva_np, val_proba)
    print(f"✅ Best threshold (F1): {thr:.2f} | precision={thr_metrics['precision']:.2f} | recall={thr_metrics['recall']:.2f} | f1={thr_metrics['f1']:.2f}")

    # --- Reports ---
    _print_header(f"{model_key.upper()} — Validation Report")
    _report_block("Validation", yva_np, val_proba, thr)

    _print_header(f"{model_key.upper()} — Test Report")
    _report_block("Test", yte_np, test_proba, thr)

    # --- Save artifacts ---
    extras: Dict[str, Any] = {"val_auc": val_auc, "val_pr_auc": pr_auc, "best_thr": thr, **thr_metrics}
    if save_dir is not None:
        out_dir = save_dir / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # metrics json
        with open(out_dir / "val_metrics.json", "w") as f:
            json.dump({"roc_auc": val_auc, "pr_auc": pr_auc, "best_thr": thr, **thr_metrics}, f, indent=2)

        # classification reports as text
        def _report_to_txt(y_true, y_scores, thr, path):
            y_pred = (y_scores >= thr).astype(int)
            cr = classification_report(y_true, y_pred, digits=4)
            with open(path, "w") as fh:
                fh.write(cr)
        _report_to_txt(yva_np, val_proba, thr, out_dir / "validation_report.txt")
        _report_to_txt(yte_np, test_proba, thr, out_dir / "test_report.txt")

        # confusion matrices
        cm_val = confusion_matrix(yva_np, (val_proba >= thr).astype(int))
        cm_te  = confusion_matrix(yte_np, (test_proba >= thr).astype(int))
        np.savetxt(out_dir / "val_confusion_matrix.csv", cm_val, delimiter=",", fmt="%d")
        np.savetxt(out_dir / "test_confusion_matrix.csv", cm_te, delimiter=",", fmt="%d")

    return ModelResult(
        model=model,
        val_auc=val_auc,
        val_pr_auc=pr_auc,
        best_thr=thr,
        best_thr_metrics=thr_metrics,
        val_proba=val_proba,
        test_proba=test_proba,
        extras=extras,
    )


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------

def run_all(
    *,
    X_train_res_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    X_train_res: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_res: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    models=("lr", "rf", "xgb", "mlp", "tabnet"),
    random_state: int = 42,
    save_dir: Optional[str] = None,
) -> Dict[str, ModelResult]:
    """
    Train/evaluate the requested models and print plots + reports.

    If save_dir is provided, artifacts are written under save_dir/<model>/
    Returns: dict {model_key -> ModelResult}
    """
    results: Dict[str, ModelResult] = {}
    save_path = Path(save_dir) if save_dir is not None else None

    for key in models:
        try:
            res = fit_and_eval(
                key,
                X_train_res_scaled,
                X_val_scaled,
                X_test_scaled,
                X_train_res,
                X_val,
                X_test,
                y_train_res,
                y_val,
                y_test,
                random_state=random_state,
                save_dir=save_path,
            )
            results[key] = res
        except ImportError as e:
            _print_header(f"{key.upper()} — SKIPPED (missing dependency)")
            print(str(e))
        except Exception as e:
            _print_header(f"{key.upper()} — FAILED")
            print(repr(e))

    # Leaderboards
    if results:
        roc_board = sorted(((k, v.val_auc) for k, v in results.items()), key=lambda x: x[1], reverse=True)
        _print_header("Validation ROC AUC Leaderboard")
        for rank, (k, auc) in enumerate(roc_board, 1):
            print(f"{rank:>2}. {k.upper():<6}  ROC AUC={auc:.4f}")

        pr_board = sorted(((k, v.val_pr_auc) for k, v in results.items()), key=lambda x: x[1], reverse=True)
        _print_header("Validation PR AUC Leaderboard (positive class)")
        for rank, (k, auc) in enumerate(pr_board, 1):
            print(f"{rank:>2}. {k.upper():<6}  PR AUC={auc:.4f}")

    return results
