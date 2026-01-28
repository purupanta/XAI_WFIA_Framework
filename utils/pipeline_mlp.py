# ============================
# MLP Pipeline (train/validate/test + save) — corrected
#  - Metrics now include:
#       auc_roc (AUC-ROC), auc_pr (AUC-PR),
#       logloss, brier, acc,
#       f1 (positive-class F1), macro_f1 (macro-averaged F1), mcc
#  - In mlp_val_summary.csv / mlp_test_summary.csv:
#       F1 column is written as macro_avg_f1 (macro-averaged F1)
#       AUC renamed to auc_roc and auc_pr added
# ============================
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
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
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning


# ---------- tiny utils ----------
def _stamp(msg: str):
    print(msg)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _outdir(CONFIGS: Dict[str, Any]) -> Path:
    p = Path(CONFIGS["DIR_tr_va_te_metric_shap_SAVE_DIR"]) / "mlp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig, path: Path, dpi: int = 150):
    path = path.with_suffix(".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _stamp(f"✓ Saved figure: {path.resolve()}")


def _save_text(text: str, path: Path):
    path = path.with_suffix(".txt")
    path.write_text(text, encoding="utf-8")
    _stamp(f"✓ Saved text:   {path.resolve()}")


def _save_csv(df: pd.DataFrame, path: Path):
    path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    _stamp(f"✓ Saved CSV:    {path.resolve()}")


# ---------- core helpers ----------
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


def _balanced_sample_weight(y):
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)
    w_pos = 0.5 / pos
    w_neg = 0.5 / neg
    sw = np.where(y == 1, w_pos, w_neg)
    return (sw * y.size).astype(np.float32)


def _supports_sample_weight(estimator_fit) -> bool:
    import inspect

    try:
        sig = inspect.signature(estimator_fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False


def _metrics(y_true, p, t: float = 0.5) -> Dict[str, float]:
    """
    Extended metrics for model comparison.

    Returns:
        dict with keys:
          - auc_roc  : ROC AUC
          - auc_pr   : PR AUC (average precision)
          - logloss
          - brier
          - acc
          - f1       : positive-class F1
          - macro_f1 : macro-averaged F1
          - mcc      : Matthews correlation coefficient
    """
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p).reshape(-1)
    y_pred = (p1 >= t).astype(int)

    auc_roc = roc_auc_score(y_true, p1)
    auc_pr = average_precision_score(y_true, p1)
    ll = log_loss(y_true, p1, labels=[0, 1])
    brier = brier_score_loss(y_true, p1)
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

    out = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "logloss": ll,
        "brier": brier,
        "acc": acc,
        "f1": f1_pos,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }
    return {k: float(v) for k, v in out.items()}


def _metrics_split_to_df(split_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, md in split_dict.items():
        row = {"model": name}
        row.update(md)
        rows.append(row)
    return pd.DataFrame(rows)


def _optimize_threshold(y_true, p, *, metric: str = "f1") -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)
    grid = np.linspace(0.05, 0.95, 19)

    if metric.lower() == "youden":
        from sklearn.metrics import recall_score

        def score_at(t):
            yhat = (p >= t).astype(int)
            tn = np.sum((y_true == 0) & (yhat == 0))
            fp = np.sum((y_true == 0) & (yhat == 1))
            spec = tn / (tn + fp + 1e-9)
            sens = recall_score(y_true, yhat)
            return sens + spec - 1.0

    else:
        def score_at(t):
            return f1_score(y_true, (p >= t).astype(int))

    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = score_at(float(t))
        if s > best_s:
            best_s, best_t = s, float(t)
    _stamp(f"[MLP][THRESH] best_t={best_t:.3f} ({metric}={best_s:.4f})")
    return float(best_t)


# ---------- isotonic calibrator wrapper ----------
class _IsotonicCalibrated:
    def __init__(self, base_estimator, iso: IsotonicRegression):
        self.base_estimator = base_estimator
        self.iso = iso
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        p = _predict_proba_safe(self.base_estimator, X)[:, 1]
        pc = np.clip(self.iso.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------- feature importance (Garson-like path aggregation) ----------
def _mlp_feature_importance(
    mlp: MLPClassifier,
    feature_names: List[str],
    X_val=None,
    p_val=None,
    topn: Optional[int] = 30,
) -> pd.DataFrame:
    if not hasattr(mlp, "coefs_") or len(mlp.coefs_) == 0:
        return pd.DataFrame(
            columns=["feature", "importance", "abs_importance", "sign"]
        )
    coefs = [np.asarray(w) for w in mlp.coefs_]
    # Start from output layer: reduce to one vector of size last hidden
    contrib = np.abs(coefs[-1]).sum(axis=1)  # (hidden_last,)
    # Propagate back to inputs
    for L in range(len(coefs) - 2, -1, -1):
        contrib = np.abs(coefs[L]) @ contrib  # final L=0 -> (n_features,)
    imp = contrib.astype(float)

    # Optional sign via correlation with validation probs
    if X_val is not None and p_val is not None:
        try:
            xv = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
            signs = []
            for i in range(xv.shape[1]):
                col = xv[:, i]
                if np.std(col) < 1e-12:
                    signs.append(0.0)
                else:
                    c = np.corrcoef(col, np.asarray(p_val).reshape(-1))[0, 1]
                    signs.append(np.sign(c) if np.isfinite(c) else 0.0)
            sign = np.array(signs, dtype=float)
        except Exception:
            sign = np.zeros_like(imp)
    else:
        sign = np.zeros_like(imp)

    df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": imp * np.where(sign == 0, 1.0, sign),
                "abs_importance": np.abs(imp),
                "sign": sign,
            }
        )
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )

    if topn and topn > 0:
        df = df.head(int(topn)).copy()
    return df


# ---------- plots ----------
def plot_mlp_roc_pr(
    y_val, p_val, y_test=None, p_test=None, title_suffix="MLP (calibrated)"
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


def plot_mlp_calibration(
    y_val, p_val, y_test=None, p_test=None, n_bins=15, title_suffix="MLP (calibrated)"
):
    fig = plt.figure()
    frac_v, mean_v = calibration_curve(
        y_val, p_val, n_bins=n_bins, strategy="quantile"
    )
    plt.plot(mean_v, frac_v, marker="o", label="Validation")
    if y_test is not None and p_test is not None:
        frac_t, mean_t = calibration_curve(
            y_test, p_test, n_bins=n_bins, strategy="quantile"
        )
        plt.plot(mean_t, frac_t, marker="s", label="Test")
    line = np.linspace(0, 1, 100)
    plt.plot(line, line, "--", color="gray")
    plt.title(f"Reliability Diagram — {title_suffix}")
    plt.xlabel("Mean predicted prob")
    plt.ylabel("Fraction of positives")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_mlp_loss(history: Dict[str, List[float]]):
    if not history:
        return None
    fig = plt.figure()
    if "loss" in history and history["loss"]:
        plt.plot(
            range(1, len(history["loss"]) + 1),
            history["loss"],
            label="train loss",
        )
    if "val_score" in history and history["val_score"]:
        plt.plot(
            range(1, len(history["val_score"]) + 1),
            history["val_score"],
            label="internal val score",
        )
    if "val_logloss_once" in history:
        plt.axhline(
            history["val_logloss_once"],
            linestyle="--",
            label=f"val_logloss_once={history['val_logloss_once']:.3f}",
        )
    plt.title("MLP: training loss / internal val")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_top_features_bar(df_top: pd.DataFrame, title="MLP Top Features"):
    if df_top is None or df_top.empty:
        return None
    fig = plt.figure(figsize=(8, max(5, 0.32 * len(df_top))))
    y = np.arange(len(df_top))
    vals = df_top["abs_importance"].values
    plt.barh(y, vals)
    plt.yticks(y, df_top["feature"].values)
    plt.gca().invert_yaxis()  # largest at top
    for i, v in enumerate(vals):
        plt.text(v, i, f" {v:.3g}", va="center")
    plt.xlabel("|importance|")
    plt.title(title)
    plt.tight_layout()
    return fig


# ---------- summary helper for *_summary.csv ----------
def _summary_df(md: Dict[str, float], threshold: float) -> pd.DataFrame:
    """
    Build summary row with renamed metrics:
      - auc_roc, auc_pr, logloss, brier, acc, macro_avg_f1, mcc
    """
    cols = [
        "threshold",
        "auc_roc",
        "auc_pr",
        "logloss",
        "brier",
        "acc",
        "macro_avg_f1",
        "mcc",
    ]
    row = {
        "threshold": float(threshold),
        "auc_roc": md.get("auc_roc", np.nan),
        "auc_pr": md.get("auc_pr", np.nan),
        "logloss": md.get("logloss", np.nan),
        "brier": md.get("brier", np.nan),
        "acc": md.get("acc", np.nan),
        # use macro_f1 if present; fall back to f1 if not
        "macro_avg_f1": md.get("macro_f1", md.get("f1", np.nan)),
        "mcc": md.get("mcc", np.nan),
    }
    return pd.DataFrame([row], columns=cols)


# ---------- main pipeline ----------
def train_validate_test_mlp(
    X_train_res_scaled,
    y_train_res,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test,
    feature_names: Optional[List[str]] = None,
    *,
    hidden_grid: Optional[List[Tuple[int, ...]]] = None,
    alpha_grid: Optional[List[float]] = None,
    learning_rate_init: float = 8e-4,
    max_iter: int = 400,
    random_state: int = 42,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,  # 0/None => ALL
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Full MLP pipeline on *scaled* splits with isotonic calibration and thresholding.

    Outputs (under CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] / 'mlp'):

      CSV:
        - mlp_metrics_train.csv / mlp_metrics_val.csv / mlp_metrics_test.csv
          columns include: auc_roc, auc_pr, logloss, brier, acc, f1, macro_f1, mcc
        - mlp_val_summary.csv / mlp_test_summary.csv
          columns: threshold, auc_roc, auc_pr, logloss, brier, acc, macro_avg_f1, mcc

      Text:
        - mlp_val_report.txt / mlp_test_report.txt

      Plots:
        - mlp_roc.png
        - mlp_pr.png
        - mlp_calibration.png
        - mlp_loss_curve.png
        - mlp_top_features_bar.png

      Feature importance:
        - mlp_top{K}.csv (Garson-like, by |importance|)
    """
    assert (
        CONFIGS is not None
        and "DIR_tr_va_te_metric_shap_SAVE_DIR" in CONFIGS
    ), "CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] required"

    outdir = _outdir(CONFIGS) if save_outputs else None
    if outdir is not None:
        _stamp(f"[SAVE] Output directory: {outdir.resolve()}")

    feature_names = feature_names or list(
        getattr(
            X_train_res_scaled,
            "columns",
            [f"f{i}" for i in range(X_train_res_scaled.shape[1])],
        )
    )

    # ---- tiny hyperparam grid (fast) ----
    hidden_grid = hidden_grid or [(128, 64), (128,), (64,), (256, 128)]
    alpha_grid = alpha_grid or [1e-4, 5e-4]

    best = None
    best_hist = {}
    sw = _balanced_sample_weight(y_train_res)
    can_sw = _supports_sample_weight(MLPClassifier.fit)

    _stamp("[MLP] Hyperparam scan:")
    for h in hidden_grid:
        for a in alpha_grid:
            mlp = MLPClassifier(
                hidden_layer_sizes=h,
                activation="relu",
                solver="adam",
                alpha=float(a),
                learning_rate_init=float(learning_rate_init),
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.15,
                max_iter=int(max_iter),
                random_state=int(random_state),
                warm_start=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                if can_sw:
                    mlp.fit(X_train_res_scaled, y_train_res, sample_weight=sw)
                else:
                    mlp.fit(X_train_res_scaled, y_train_res)

            p_val = _predict_proba_safe(mlp, X_val_scaled)[:, 1]
            v_ll = log_loss(y_val, p_val, labels=[0, 1])
            v_auc = roc_auc_score(y_val, p_val)
            _stamp(
                f"  h={h}  alpha={a}  val_logloss={v_ll:.4f}  val_auc={v_auc:.4f}"
            )

            if (
                (best is None)
                or (v_ll < best["val_logloss"] - 1e-9)
                or (
                    abs(v_ll - best["val_logloss"]) < 1e-9
                    and v_auc > best["val_auc"]
                )
            ):
                best = {
                    "h": h,
                    "alpha": a,
                    "model": mlp,
                    "val_logloss": v_ll,
                    "val_auc": v_auc,
                }
                hist = {"loss": list(getattr(mlp, "loss_curve_", []))}
                if hasattr(mlp, "validation_scores_"):
                    hist["val_score"] = list(mlp.validation_scores_)
                try:
                    hist["val_logloss_once"] = float(v_ll)
                except Exception:
                    pass
                best_hist = hist

    mlp_raw = best["model"]
    _stamp(
        f"[MLP] Best hidden={best['h']}, alpha={best['alpha']} selected by val_logloss (tie-break AUC)."
    )

    # ---- isotonic calibration on validation ----
    p_val_uncal = _predict_proba_safe(mlp_raw, X_val_scaled)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        p_val_uncal, np.asarray(y_val).astype(int)
    )
    mlp_cal = _IsotonicCalibrated(mlp_raw, iso)
    _stamp("[MLP] Isotonic calibration done on validation.")

    # ---- threshold on validation ----
    p_val_cal = _predict_proba_safe(mlp_cal, X_val_scaled)[:, 1]
    best_t = _optimize_threshold(y_val, p_val_cal, metric=threshold_metric)

    # ---- metrics across splits ----
    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # raw
    p_tr_raw = _predict_proba_safe(mlp_raw, X_train_res_scaled)[:, 1]
    p_va_raw = p_val_uncal
    p_te_raw = (
        _predict_proba_safe(mlp_raw, X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["mlp_raw@0.50"] = _metrics(
        y_train_res, p_tr_raw, 0.5
    )
    out_metrics["val"]["mlp_raw@0.50"] = _metrics(y_val, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["mlp_raw@0.50"] = _metrics(y_test, p_te_raw, 0.5)

    # calibrated
    p_tr_cal = _predict_proba_safe(mlp_cal, X_train_res_scaled)[:, 1]
    p_te_cal = (
        _predict_proba_safe(mlp_cal, X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["mlp_cal@0.50"] = _metrics(
        y_train_res, p_tr_cal, 0.5
    )
    out_metrics["val"]["mlp_cal@0.50"] = _metrics(y_val, p_val_cal, 0.5)
    out_metrics["val"][f"mlp_cal@{best_t:.2f}"] = _metrics(
        y_val, p_val_cal, best_t
    )
    if p_te_cal is not None:
        out_metrics["test"]["mlp_cal@0.50"] = _metrics(
            y_test, p_te_cal, 0.5
        )
        out_metrics["test"][f"mlp_cal@{best_t:.2f}"] = _metrics(
            y_test, p_te_cal, best_t
        )

    # ---- reports (Val + Test) ----
    val_md = out_metrics["val"].get(f"mlp_cal@{best_t:.2f}") or _metrics(
        y_val, p_val_cal, best_t
    )
    y_val_pred = (p_val_cal >= best_t).astype(int)
    val_report_text = (
        f"Best threshold (F1 on VAL): {best_t:.2f}\n\n"
        f"Validation metrics (calibrated, threshold = {best_t:.2f}):\n"
        f"  Accuracy       : {val_md.get('acc', np.nan):.3f}\n"
        f"  AUC-ROC        : {val_md.get('auc_roc', np.nan):.3f}\n"
        f"  AUC-PR         : {val_md.get('auc_pr', np.nan):.3f}\n"
        f"  Brier          : {val_md.get('brier', np.nan):.4f}\n"
        f"  MCC            : {val_md.get('mcc', np.nan):.3f}\n"
        f"  F1 (macro_avg) : {val_md.get('macro_f1', np.nan):.3f}\n\n"
        f"Classification Report (Validation):\n"
        f"{classification_report(y_val, y_val_pred, target_names=['Class 0','Class 1'], digits=2)}"
    )
    print("\n" + val_report_text)

    test_report_text = ""
    test_md: Optional[Dict[str, float]] = None
    if X_test_scaled is not None and p_te_cal is not None:
        test_md = out_metrics["test"].get(
            f"mlp_cal@{best_t:.2f}"
        ) or _metrics(y_test, p_te_cal, best_t)
        y_pred_test = (p_te_cal >= best_t).astype(int)
        test_report_text = (
            f"Best threshold (F1): {best_t:.2f}\n\n"
            f"Test metrics (calibrated, threshold = {best_t:.2f}):\n"
            f"  Accuracy       : {test_md.get('acc', np.nan):.3f}\n"
            f"  AUC-ROC        : {test_md.get('auc_roc', np.nan):.3f}\n"
            f"  AUC-PR         : {test_md.get('auc_pr', np.nan):.3f}\n"
            f"  Brier          : {test_md.get('brier', np.nan):.4f}\n"
            f"  MCC            : {test_md.get('mcc', np.nan):.3f}\n"
            f"  F1 (macro_avg) : {test_md.get('macro_f1', np.nan):.3f}\n\n"
            f"Classification Report (Test):\n"
            f"{classification_report(y_test, y_pred_test, target_names=['Class 0','Class 1'], digits=2)}"
        )
        print("\n" + test_report_text)

    # ---- feature importance (Garson-like) ----
    topn = None if (topn_features is None or topn_features == 0) else int(
        topn_features
    )
    feat_df = _mlp_feature_importance(
        mlp_raw,
        feature_names,
        X_val=X_val_scaled,
        p_val=p_val_cal,
        topn=topn,
    )

    # ---- plots ----
    loss_fig = plot_mlp_loss(best_hist)
    roc_fig, pr_fig = plot_mlp_roc_pr(
        y_val,
        p_val_cal,
        y_test if X_test_scaled is not None else None,
        p_te_cal if X_test_scaled is not None else None,
        title_suffix="MLP (Calibrated)",
    )
    calib_fig = plot_mlp_calibration(
        y_val,
        p_val_cal,
        y_test if X_test_scaled is not None else None,
        p_te_cal if X_test_scaled is not None else None,
        title_suffix="MLP (Calibrated)",
    )
    top_bar_fig = plot_top_features_bar(feat_df)

    # ---- summaries (CSV rows for threshold=best_t) ----
    val_summary_df = _summary_df(val_md, best_t)

    test_summary_df = pd.DataFrame()
    if X_test_scaled is not None and p_te_cal is not None and test_md is not None:
        test_summary_df = _summary_df(test_md, best_t)

    # ---- save everything ----
    if outdir is not None:
        if out_metrics["train"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["train"]),
                outdir / "mlp_metrics_train",
            )
        if out_metrics["val"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["val"]),
                outdir / "mlp_metrics_val",
            )
        if out_metrics["test"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["test"]),
                outdir / "mlp_metrics_test",
            )

        _save_text(val_report_text, outdir / "mlp_val_report")
        if test_report_text:
            _save_text(test_report_text, outdir / "mlp_test_report")

        _save_csv(val_summary_df, outdir / "mlp_val_summary")
        if not test_summary_df.empty:
            _save_csv(test_summary_df, outdir / "mlp_test_summary")

        _save_csv(
            feat_df,
            outdir
            / (
                f"mlp_top{len(feat_df)}"
                if topn is not None
                else "mlp_topALL"
            ),
        )

        _save_fig(roc_fig, outdir / "mlp_roc")
        _save_fig(pr_fig, outdir / "mlp_pr")
        _save_fig(calib_fig, outdir / "mlp_calibration")
        if loss_fig is not None:
            _save_fig(loss_fig, outdir / "mlp_loss_curve")
        if top_bar_fig is not None:
            _save_fig(top_bar_fig, outdir / "mlp_top_features_bar")

    return {
        "mlp_raw": mlp_raw,
        "mlp_cal": mlp_cal,
        "best_hidden": best["h"],
        "best_alpha": best["alpha"],
        "best_threshold": best_t,
        "metrics": out_metrics,
        "history": best_hist,
        "features": feat_df,
        "val_report_text": val_report_text,
        "test_report_text": test_report_text,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
        "outdir": (str(outdir) if outdir is not None else None),
    }
