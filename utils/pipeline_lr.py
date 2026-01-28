# ============================
# Logistic Regression Pipeline (with saving) — CSV summaries
# ============================
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
    accuracy_score,
    f1_score,
    brier_score_loss,
    classification_report,
    matthews_corrcoef,
)
from sklearn.exceptions import ConvergenceWarning
import warnings


# ---- Isotonic wrapper (no cv='prefit' deprecation) ----
class _IsotonicCalibrated:
    """Wraps a prefit classifier with an isotonic calibrator fitted on validation probs."""
    def __init__(self, base_estimator, iso):
        import numpy as np
        self.base_estimator = base_estimator
        self.iso = iso
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        import numpy as np
        p = _predict_proba_safe(self.base_estimator, X)[:, 1]
        pc = np.clip(self.iso.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------- basic io helpers ----------
def _stamp(msg: str):
    print(msg)


def _save_fig(fig, path: Path, dpi: int = 150):
    path = path.with_suffix(".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_text(text: str, path: Path):
    path = path.with_suffix(".txt")
    path.write_text(text, encoding="utf-8")


def _save_csv(df: pd.DataFrame, path: Path):
    path = path.with_suffix(".csv")
    df.to_csv(path, index=False)


def _outdir(CONFIGS: Dict[str, Any]) -> Path:
    p = Path(CONFIGS["DIR_tr_va_te_metric_shap_SAVE_DIR"]) / "lr"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- model/metric helpers ----------
def _predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        elif p.shape[1] == 1:
            p = np.column_stack([1 - p[:, 0], p[:, 0]])
        return p
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X)).reshape(-1)
        prob1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - prob1, prob1])
    raise AttributeError("Model does not support probability predictions.")


def _metrics(y_true, p, t: float = 0.5) -> Dict[str, float]:
    """
    Compute threshold-dependent and threshold-independent metrics.

    Returns
    -------
    Dict[str, float]
        {
          'auc_roc' : AUC-ROC,
          'auc_pr'  : AUC-PR (average precision),
          'logloss',
          'brier',
          'acc',
          'f1'      : positive-class F1,
          'macro_f1': macro-averaged F1,
          'mcc'
        }
    """
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p).reshape(-1)
    y_pred = (p1 >= t).astype(int)

    auc_roc = roc_auc_score(y_true, p1)
    auc_pr = average_precision_score(y_true, p1)
    logloss = log_loss(y_true, p1, labels=[0, 1])
    brier = brier_score_loss(y_true, p1)
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)                    # positive-class F1
    macro_f1 = f1_score(y_true, y_pred, average="macro") # macro F1
    mcc = matthews_corrcoef(y_true, y_pred)

    out = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "logloss": logloss,
        "brier": brier,
        "acc": acc,
        "f1": f1_pos,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }
    return {k: float(v) for k, v in out.items()}


def _optimize_threshold(y_true, p, *, metric: str = "f1") -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)
    grid = np.linspace(0.05, 0.95, 19)

    if metric.lower() == "youden":
        from sklearn.metrics import recall_score

        def scorer(t):
            yhat = (p >= t).astype(int)
            tn = np.sum((y_true == 0) & (yhat == 0))
            fp = np.sum((y_true == 0) & (yhat == 1))
            spec = tn / (tn + fp + 1e-9)
            sens = recall_score(y_true, yhat)
            return sens + spec - 1.0

    else:
        # optimize positive-class F1
        def scorer(t):
            return f1_score(y_true, (p >= t).astype(int))

    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = scorer(float(t))
        if s > best_s:
            best_s, best_t = s, float(t)
    _stamp(f"[LR][THRESH] best_t={best_t:.3f} ({metric}={best_s:.4f})")
    return float(best_t)


def _feature_table(
    lr_model: LogisticRegression,
    feature_names: List[str],
    topn: Optional[int],
) -> pd.DataFrame:
    """Return features sorted by |coef|. If topn is None/<=0 or >=n, return ALL."""
    coef = lr_model.coef_.ravel()
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef),
            "odds_ratio": np.exp(coef),
        }
    ).sort_values("abs_coef", ascending=False)
    if (topn is None) or (topn <= 0) or (topn >= len(df)):
        return df.reset_index(drop=True)
    return df.head(int(topn)).reset_index(drop=True)


def _metrics_split_to_df(split_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, md in split_dict.items():
        row = {"model": name}
        row.update(md)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_df(md: Dict[str, float], threshold: float) -> pd.DataFrame:
    """
    One-row DataFrame for summaries.

    In the summary CSVs:
    - 'macro_avg_f1' column holds macro-averaged F1.
    - Full metrics CSVs still have both 'f1' (positive-class) and 'macro_f1'.
    """
    order = ["threshold", "auc_roc", "auc_pr", "logloss", "brier", "acc", "macro_avg_f1", "mcc"]
    row = {
        "threshold": float(threshold),
        "auc_roc": md.get("auc_roc", np.nan),
        "auc_pr": md.get("auc_pr", np.nan),
        "logloss": md.get("logloss", np.nan),
        "brier": md.get("brier", np.nan),
        "acc": md.get("acc", np.nan),
        "macro_avg_f1": md.get("macro_f1", md.get("f1", np.nan)),
        "mcc": md.get("mcc", np.nan),
    }
    return pd.DataFrame([row], columns=order)


# ---------- plots ----------
def plot_lr_roc_pr(
    y_val,
    p_val,
    y_test,
    p_test,
    title_suffix: str = "LR (calibrated)",
):
    figs = []

    # ROC
    fig = plt.figure()
    fpr_v, tpr_v, _ = roc_curve(y_val, p_val)
    auc_v = roc_auc_score(y_val, p_val)
    plt.plot(fpr_v, tpr_v, label=f"Val (AUC-ROC={auc_v:.3f})")
    if y_test is not None and p_test is not None:
        fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
        auc_t = roc_auc_score(y_test, p_test)
        plt.plot(fpr_t, tpr_t, label=f"Test (AUC-ROC={auc_t:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.title(f"ROC — {title_suffix}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    figs.append(fig)

    # PR
    fig = plt.figure()
    prec_v, rec_v, _ = precision_recall_curve(y_val, p_val)
    ap_v = average_precision_score(y_val, p_val)
    plt.plot(rec_v, prec_v, label=f"Val (AUC-PR={ap_v:.3f})")
    if y_test is not None and p_test is not None:
        prec_t, rec_t, _ = precision_recall_curve(y_test, p_test)
        ap_t = average_precision_score(y_test, p_test)
        plt.plot(rec_t, prec_t, label=f"Test (AUC-PR={ap_t:.3f})")
    plt.title(f"Precision–Recall — {title_suffix}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    figs.append(fig)

    return figs  # [roc_fig, pr_fig]


def plot_lr_calibration(
    y_val,
    p_val,
    y_test=None,
    p_test=None,
    n_bins: int = 15,
    title_suffix: str = "LR (calibrated)",
):
    fig = plt.figure()
    frac_v, mean_v = calibration_curve(y_val, p_val, n_bins=n_bins, strategy="quantile")
    plt.plot(mean_v, frac_v, marker="o", label="Validation")
    if y_test is not None and p_test is not None:
        frac_t, mean_t = calibration_curve(
            y_test, p_test, n_bins=n_bins, strategy="quantile"
        )
        plt.plot(mean_t, frac_t, marker="s", label="Test")
    line = np.linspace(0, 1, 100)
    plt.plot(line, line, "--")
    plt.title(f"Reliability Diagram — {title_suffix}")
    plt.xlabel("Mean predicted prob")
    plt.ylabel("Fraction of positives")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_lr_loss_curve(curve: Dict[str, List[float]]):
    if not curve:
        return None
    it = curve.get("iter", [])
    tr = curve.get("train_logloss", [])
    va = curve.get("val_logloss", [])
    if not it or (not tr and not va):
        return None

    fig = plt.figure()
    if tr:
        plt.plot(it[: len(tr)], tr, label="train logloss")
    if va:
        plt.plot(it[: len(va)], va, label="val logloss")
    plt.title("LR: logloss vs iterations (warm_start simulation)")
    plt.xlabel("max_iter")
    plt.ylabel("logloss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_features_bar(
    df_top: pd.DataFrame,
    title: str = "LR Features (|coef|)",
    label_col: str = "abs_coef",
    sort_desc: bool = True,
):
    """
    Horizontal bar chart of features with values annotated at bar ends.
    """
    if df_top is None or df_top.empty or label_col not in df_top.columns:
        return None

    dfp = df_top.copy()
    if sort_desc:
        dfp = dfp.sort_values(label_col, ascending=False).reset_index(drop=True)

    values = dfp[label_col].to_numpy()
    labels = dfp["feature"].astype(str).to_numpy()
    n = len(dfp)

    fig = plt.figure(figsize=(8, max(5, 0.28 * n)))
    ax = plt.gca()

    y = np.arange(n)
    bars = ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    vmax = float(np.nanmax(values)) if n else 1.0
    ax.set_xlim(0, vmax * 1.12 if vmax > 0 else 1.0)

    offset = vmax * 0.01 if vmax > 0 else 0.02
    for bar, val in zip(bars, values):
        x = bar.get_width()
        ytxt = bar.get_y() + bar.get_height() / 2
        ax.text(x + offset, ytxt, f"{val:.3f}", va="center", ha="left", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlabel(label_col)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_lr_brier_mcc(
    val_md: Dict[str, float],
    test_md: Optional[Dict[str, float]] = None,
    title_suffix: str = "Logistic Regression (Calibrated)",
):
    """
    Bar plot for Brier score and MCC on validation (and test, if provided).
    """
    metrics = ["brier", "mcc"]
    labels = [m.upper() for m in metrics]
    val_vals = [float(val_md.get(m, np.nan)) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig = plt.figure()
    ax = plt.gca()

    if test_md is not None:
        ax.bar(x - width / 2, val_vals, width, label="Validation")
    else:
        ax.bar(x, val_vals, width * 1.2, label="Validation")

    if test_md is not None:
        test_vals = [float(test_md.get(m, np.nan)) for m in metrics]
        ax.bar(x + width / 2, test_vals, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Brier & MCC — {title_suffix}")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y")
    ax.legend()

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


# ---------- main pipeline ----------
def train_validate_test_lr(
    X_train_res_scaled,
    y_train_res,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test,
    feature_names: Optional[List[str]] = None,
    *,
    C_grid: List[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    penalty: str = "l2",
    prefer_saga: bool = True,
    max_iter: int = 600,
    random_state: int = 42,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Saves to: CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] / 'lr'
    Creates:
      - lr_metrics_train.csv, lr_metrics_val.csv, lr_metrics_test.csv
      - lr_features_top{K}.csv or lr_features_all.csv (+ matching *_bar.png)
      - lr_roc.png, lr_pr.png, lr_calibration.png, lr_loss_curve.png
      - lr_brier_mcc.png
      - lr_val_report.txt, lr_test_report.txt
      - lr_val_summary.csv, lr_test_summary.csv  (with macro_avg_f1 column)
      - lr_brier_mcc_summary.csv
      - lr_summary.txt  (best C & best threshold)
    """
    CONFIGS = CONFIGS or {}
    outdir = _outdir(CONFIGS) if save_outputs else None
    if outdir is not None:
        _stamp(f"[SAVE] Output directory: {outdir.resolve()}")

    # Allow CONFIGS override of top-K
    if "LR_TOPN_FEATURES" in CONFIGS and CONFIGS["LR_TOPN_FEATURES"] is not None:
        topn_features = CONFIGS["LR_TOPN_FEATURES"]

    # ---- 1) prep ----
    feature_names = feature_names or list(
        getattr(
            X_train_res_scaled,
            "columns",
            [f"f{i}" for i in range(X_train_res_scaled.shape[1])],
        )
    )

    # ---- 2) small grid over C on validation ----
    best = None
    _stamp("[LR] Hyperparam scan (C):")
    for C in C_grid:
        solver = "saga" if (prefer_saga or penalty in ("l1", "elasticnet")) else "lbfgs"
        l1_ratio = 0.15 if penalty == "elasticnet" else None
        lr = LogisticRegression(
            C=float(C),
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(random_state),
            n_jobs=-1 if solver == "saga" else None,
            warm_start=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lr.fit(X_train_res_scaled, y_train_res)

        p_val = _predict_proba_safe(lr, X_val_scaled)[:, 1]
        v_ll = log_loss(y_val, p_val, labels=[0, 1])
        v_auc = roc_auc_score(y_val, p_val)

        _stamp(f"  C={C:<6}  val_logloss={v_ll:.4f}  val_auc_roc={v_auc:.4f}")

        if (
            best is None
            or (v_ll < best["val_logloss"] - 1e-9)
            or (abs(v_ll - best["val_logloss"]) < 1e-9 and v_auc > best["val_auc_roc"])
        ):
            best = {"C": C, "model": lr, "val_logloss": v_ll, "val_auc_roc": v_auc}

    lr_raw = best["model"]
    _stamp(f"[LR] Best C={best['C']} selected by val_logloss (tie-break AUC-ROC).")

    # ---- 3) loss curve by warm-start (optional) ----
    curve = {"iter": [], "train_logloss": [], "val_logloss": []}
    if prefer_saga:
        lr_ws = LogisticRegression(
            C=float(best["C"]),
            penalty=penalty,
            solver="saga",
            l1_ratio=(0.15 if penalty == "elasticnet" else None),
            class_weight="balanced",
            max_iter=1,
            random_state=int(random_state),
            n_jobs=-1,
            warm_start=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for k in range(20, max_iter + 1, 20):
                lr_ws.set_params(max_iter=int(k))
                lr_ws.fit(X_train_res_scaled, y_train_res)
                p_tr = _predict_proba_safe(lr_ws, X_train_res_scaled)[:, 1]
                p_va = _predict_proba_safe(lr_ws, X_val_scaled)[:, 1]
                curve["iter"].append(k)
                curve["train_logloss"].append(
                    log_loss(y_train_res, p_tr, labels=[0, 1])
                )
                curve["val_logloss"].append(
                    log_loss(y_val, p_va, labels=[0, 1])
                )
    else:
        _stamp("[LR] Loss curve skipped (prefer_saga=False).")

    # ---- 4) isotonic calibration on VALIDATION (no cv='prefit') ----
    p_val_uncal = _predict_proba_safe(lr_raw, X_val_scaled)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        p_val_uncal, np.asarray(y_val).astype(int)
    )
    lr_cal = _IsotonicCalibrated(lr_raw, iso)
    _stamp("[LR] Isotonic calibration done via manual isotonic regression on validation.")

    # ---- 5) threshold tuning on VALIDATION ----
    p_val_cal = _predict_proba_safe(lr_cal, X_val_scaled)[:, 1]
    best_t = _optimize_threshold(y_val, p_val_cal, metric=threshold_metric)

    # ---- 6) metrics on TRAIN / VAL / TEST ----
    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # Raw LR
    p_tr_raw = _predict_proba_safe(lr_raw, X_train_res_scaled)[:, 1]
    p_va_raw = _predict_proba_safe(lr_raw, X_val_scaled)[:, 1]
    p_te_raw = (
        _predict_proba_safe(lr_raw, X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["lr_raw@0.50"] = _metrics(y_train_res, p_tr_raw, 0.5)
    out_metrics["val"]["lr_raw@0.50"] = _metrics(y_val, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["lr_raw@0.50"] = _metrics(y_test, p_te_raw, 0.5)

    # Calibrated LR
    p_tr_cal = _predict_proba_safe(lr_cal, X_train_res_scaled)[:, 1]
    try:
        p_te_cal = (
            _predict_proba_safe(lr_cal, X_test_scaled)[:, 1]
            if (X_test_scaled is not None)
            else None
        )
    except Exception:
        p_te_cal = None  # ensure test summary/report still handled

    out_metrics["train"]["lr_cal@0.50"] = _metrics(y_train_res, p_tr_cal, 0.5)
    out_metrics["val"]["lr_cal@0.50"] = _metrics(y_val, p_val_cal, 0.5)
    out_metrics["val"][f"lr_cal@{best_t:.2f}"] = _metrics(y_val, p_val_cal, best_t)
    if p_te_cal is not None:
        out_metrics["test"]["lr_cal@0.50"] = _metrics(y_test, p_te_cal, 0.5)
        out_metrics["test"][f"lr_cal@{best_t:.2f}"] = _metrics(
            y_test, p_te_cal, best_t
        )

    # ---- 7) Validation & Test classification reports (PRINT + SAVE ONCE) ----
    val_key = f"lr_cal@{best_t:.2f}"
    val_md = out_metrics["val"].get(val_key) or _metrics(y_val, p_val_cal, best_t)

    # Validation predictions at best threshold
    y_val_pred = (p_val_cal >= best_t).astype(int)

    val_report_text = (
        f"Best threshold (optimized on {threshold_metric.upper()} using positive-class F1): {best_t:.2f}\n\n"
        f"Validation metrics (calibrated, threshold = {best_t:.2f}):\n"
        f"  Accuracy       : {val_md.get('acc', np.nan):.3f}\n"
        f"  AUC-ROC        : {val_md.get('auc_roc', np.nan):.3f}\n"
        f"  AUC-PR         : {val_md.get('auc_pr', np.nan):.3f}\n"
        f"  Brier          : {val_md.get('brier', np.nan):.4f}\n"
        f"  MCC            : {val_md.get('mcc', np.nan):.3f}\n"
        f"  F1 (macro_avg) : {val_md.get('macro_f1', np.nan):.3f}\n\n"
        f"Classification Report (Validation):\n"
        f"{classification_report(y_val, y_val_pred, target_names=['Class 0', 'Class 1'], digits=2)}"
    )
    print("\n" + val_report_text)

    have_test = (X_test_scaled is not None) and (y_test is not None)
    test_report_text = ""
    test_md: Optional[Dict[str, float]] = None

    if have_test:
        if p_te_cal is not None:
            test_key = f"lr_cal@{best_t:.2f}"
            test_md = out_metrics["test"].get(test_key) or _metrics(
                y_test, p_te_cal, best_t
            )
            y_pred_test = (p_te_cal >= best_t).astype(int)

            test_report_text = (
                f"Best threshold (same as VAL): {best_t:.2f}\n\n"
                f"Test metrics (calibrated, threshold = {best_t:.2f}):\n"
                f"  Accuracy       : {test_md.get('acc', np.nan):.3f}\n"
                f"  AUC-ROC        : {test_md.get('auc_roc', np.nan):.3f}\n"
                f"  AUC-PR         : {test_md.get('auc_pr', np.nan):.3f}\n"
                f"  Brier          : {test_md.get('brier', np.nan):.4f}\n"
                f"  MCC            : {test_md.get('mcc', np.nan):.3f}\n"
                f"  F1 (macro_avg) : {test_md.get('macro_f1', np.nan):.3f}\n\n"
                f"Classification Report (Test):\n"
                f"{classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1'], digits=2)}"
            )
        else:
            test_report_text = (
                "Test report could not be generated from calibrated probabilities.\n"
                "Reason: calibrated predict_proba unavailable.\n"
                "Suggestion: check X_test_scaled shape, calibrator, or try without calibration."
            )
        print("\n" + test_report_text)

    # ---- 8) Features table (ALL or Top-K by |coef|) ----
    try:
        feat_df = _feature_table(lr_raw, feature_names, topn_features)
    except Exception:
        feat_df = pd.DataFrame(
            columns=["feature", "coef", "abs_coef", "odds_ratio"]
        )

    used_k = (
        "all"
        if (
            topn_features is None
            or topn_features <= 0
            or topn_features >= len(feature_names)
        )
        else int(topn_features)
    )
    feat_title = (
        f"LR Features (top {used_k})" if used_k != "all" else "LR Features (all)"
    )
    feat_fig = plot_features_bar(feat_df, feat_title)

    # ---- 9) Plots ----
    curve_fig = plot_lr_loss_curve(curve)
    roc_fig, pr_fig = plot_lr_roc_pr(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="Logistic Regression (Calibrated)",
    )
    calib_fig = plot_lr_calibration(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="Logistic Regression (Calibrated)",
    )

    # ---- 10) Build concise summaries (as DataFrames for CSV) ----
    def _pretty_md(md_src: Dict[str, float], thr: float) -> pd.DataFrame:
        return _summary_df(md_src, thr)

    val_summary_df = _pretty_md(val_md, best_t)

    test_summary_df = pd.DataFrame()
    if have_test:
        if test_md is None and p_te_cal is not None:
            test_md = _metrics(y_test, p_te_cal, best_t)
        if test_md is not None:
            test_summary_df = _pretty_md(test_md, best_t)
        else:
            test_summary_df = pd.DataFrame(
                [{"threshold": best_t, "note": "No calibrated probabilities for test."}]
            )

    # --- Brier & MCC figure and dedicated CSV summary ---
    brier_mcc_rows = [
        {
            "split": "val",
            "threshold": best_t,
            "brier": float(val_md.get("brier", np.nan)),
            "mcc": float(val_md.get("mcc", np.nan)),
        }
    ]

    if have_test and (test_md is not None):
        brier_mcc_rows.append(
            {
                "split": "test",
                "threshold": best_t,
                "brier": float(test_md.get("brier", np.nan)),
                "mcc": float(test_md.get("mcc", np.nan)),
            }
        )

    brier_mcc_df = pd.DataFrame(brier_mcc_rows)

    brier_mcc_fig = plot_lr_brier_mcc(
        val_md,
        test_md if (have_test and test_md is not None) else None,
        title_suffix="Logistic Regression (Calibrated)",
    )

    # ---- 11) SAVE ALL OUTPUTS ----
    if outdir is not None:
        # CSVs (full metrics tables; include f1 and macro_f1 separately)
        if out_metrics["train"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["train"]),
                outdir / "lr_metrics_train",
            )
        if out_metrics["val"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["val"]),
                outdir / "lr_metrics_val",
            )
        if out_metrics["test"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["test"]),
                outdir / "lr_metrics_test",
            )

        # Brier & MCC summary CSV
        _save_csv(brier_mcc_df, outdir / "lr_brier_mcc_summary")

        # Feature CSV + bar
        feat_suffix = f"top{used_k}" if used_k != "all" else "all"
        _save_csv(feat_df, outdir / f"lr_features_{feat_suffix}")
        if feat_fig is not None:
            _save_fig(feat_fig, outdir / f"lr_features_bar_{feat_suffix}")

        # Reports (txt)
        _save_text(val_report_text, outdir / "lr_val_report")
        if have_test:
            _save_text(test_report_text, outdir / "lr_test_report")

        # Summaries (CSV) — note: column macro_avg_f1
        _save_csv(val_summary_df, outdir / "lr_val_summary")
        if have_test:
            _save_csv(test_summary_df, outdir / "lr_test_summary")

        # Compact run summary (best C & threshold)
        _save_text(
            f"best_C={best['C']}\nbest_threshold={best_t:.3f}\n",
            outdir / "lr_summary",
        )

        # Plots
        _save_fig(roc_fig, outdir / "lr_roc")
        _save_fig(pr_fig, outdir / "lr_pr")
        _save_fig(calib_fig, outdir / "lr_calibration")
        if curve_fig is not None:
            _save_fig(curve_fig, outdir / "lr_loss_curve")
        if brier_mcc_fig is not None:
            _save_fig(brier_mcc_fig, outdir / "lr_brier_mcc")

    # ---- 12) Return everything ----
    return {
        "lr_raw": lr_raw,
        "lr_cal": lr_cal,
        "best_C": best["C"],
        "best_threshold": best_t,
        "metrics": out_metrics,
        "loss_curve": curve,
        "features": feat_df,
        "val_report_text": val_report_text,
        "test_report_text": test_report_text,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
        "brier_mcc_df": brier_mcc_df,
        "used_topn_features": used_k,
        "outdir": str(outdir) if outdir is not None else None,
    }
