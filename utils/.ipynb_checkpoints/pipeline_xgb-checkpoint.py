# ============================
# XGBoost Pipeline (with saving) — warnings fixed, metrics harmonized
# ============================
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
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

try:
    from xgboost import XGBClassifier
except Exception as _e:
    XGBClassifier = None
    _XGB_IMPORT_ERR = _e
else:
    _XGB_IMPORT_ERR = None


# ---- Isotonic wrapper (avoids cv='prefit' deprecation) ----
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
    p = Path(CONFIGS["DIR_tr_va_te_metric_shap_SAVE_DIR"]) / "xgb"
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
    _stamp(f"[XGB][THRESH] best_t={best_t:.3f} ({metric}={best_s:.4f})")
    return float(best_t)


def _xgb_feature_table(
    xgb_model: "XGBClassifier",
    feature_names: List[str],
    topn: Optional[int],
) -> pd.DataFrame:
    """
    Use 'gain' importances mapped to user feature names (handles f0,f1,... keys).
    """
    try:
        bst = xgb_model.get_booster()
        gain_map = bst.get_score(importance_type="gain")  # dict: {"f0": val, ...}
        mapped = []
        for k, v in gain_map.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                fname = feature_names[idx] if idx < len(feature_names) else k
            else:
                fname = k
            mapped.append((fname, float(v)))
        df = pd.DataFrame(mapped, columns=["feature", "importance"])
    except Exception:
        imp = getattr(xgb_model, "feature_importances_", None)
        if imp is None:
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.DataFrame({"feature": feature_names, "importance": imp})

    df = df.sort_values("importance", ascending=False)
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
def plot_xgb_roc_pr(
    y_val,
    p_val,
    y_test,
    p_test,
    title_suffix: str = "XGBoost (calibrated)",
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

    return figs


def plot_xgb_calibration(
    y_val,
    p_val,
    y_test=None,
    p_test=None,
    n_bins: int = 15,
    title_suffix: str = "XGBoost (calibrated)",
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
    plt.plot(line, line, "--", color="gray")
    plt.title(f"Reliability Diagram — {title_suffix}")
    plt.xlabel("Mean predicted prob")
    plt.ylabel("Fraction of positives")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_xgb_loss_curves(history: Dict[str, List[float]]):
    if not history or "x" not in history:
        return None
    x = history["x"]
    has_any = False
    fig = plt.figure()
    for key in ["train_logloss", "val_logloss", "train_auc_roc", "val_auc_roc"]:
        if key in history and len(history[key]) > 0:
            plt.plot(x[: len(history[key])], history[key], label=key)
            has_any = True
    if not has_any:
        return None
    plt.title("XGBoost: metrics over boosting rounds")
    plt.xlabel("Round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_features_bar(
    df_top: pd.DataFrame,
    title: str = "XGB Feature Importances (gain)",
    label_col: str = "importance",
    sort_desc: bool = True,
):
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


def plot_xgb_brier_mcc(
    val_md: Dict[str, float],
    test_md: Optional[Dict[str, float]] = None,
    title_suffix: str = "XGBoost (Calibrated)",
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
def train_validate_test_xgb(
    X_train_res_scaled,
    y_train_res,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test,
    feature_names: Optional[List[str]] = None,
    *,
    n_estimators_grid: List[int] = (600, 1200, 2000),
    learning_rate_grid: List[float] = (0.03, 0.05),
    max_depth_grid: List[int] = (4, 6, 8),
    min_child_weight_grid: List[float] = (1.0, 3.0),
    subsample_grid: List[float] = (0.8,),
    colsample_bytree_grid: List[float] = (0.7, 0.9),
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 200,
    random_state: int = 42,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,  # 0/None -> ALL
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Saves to: CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] / 'xgb'
    Creates:
      - xgb_metrics_train.csv, xgb_metrics_val.csv, xgb_metrics_test.csv
      - xgb_features_top{K}.csv or xgb_features_all.csv (+ *_bar_*.png)
      - xgb_roc.png, xgb_pr.png, xgb_calibration.png, xgb_loss_curve.png
      - xgb_brier_mcc.png, xgb_brier_mcc_summary.csv
      - xgb_val_report.txt, xgb_test_report.txt
      - xgb_val_summary.csv, xgb_test_summary.csv (with macro_avg_f1)
      - xgb_summary.txt
    """
    if XGBClassifier is None:
        raise ImportError(f"xgboost not available: {_XGB_IMPORT_ERR}")

    CONFIGS = CONFIGS or {}
    outdir = _outdir(CONFIGS) if save_outputs else None
    if outdir is not None:
        _stamp(f"[SAVE] Output directory: {outdir.resolve()}")

    if "XGB_TOPN_FEATURES" in CONFIGS and CONFIGS["XGB_TOPN_FEATURES"] is not None:
        topn_features = CONFIGS["XGB_TOPN_FEATURES"]

    # ---- 1) prep ----
    feature_names = feature_names or list(
        getattr(
            X_train_res_scaled,
            "columns",
            [f"f{i}" for i in range(X_train_res_scaled.shape[1])],
        )
    )
    y_tr = np.asarray(y_train_res).astype(int)
    pos = np.sum(y_tr == 1)
    neg = np.sum(y_tr == 0)
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0  # scale_pos_weight

    # ---- 2) small hyperparam scan with early stopping on validation ----
    best = None
    _stamp("[XGB] Hyperparam scan:")
    for n_est in n_estimators_grid:
        for lr in learning_rate_grid:
            for md in max_depth_grid:
                for mcw in min_child_weight_grid:
                    for ss in subsample_grid:
                        for cs in colsample_bytree_grid:
                            xgb = XGBClassifier(
                                objective="binary:logistic",
                                n_estimators=int(n_est),
                                learning_rate=float(lr),
                                max_depth=int(md),
                                min_child_weight=float(mcw),
                                subsample=float(ss),
                                colsample_bytree=float(cs),
                                reg_lambda=float(reg_lambda),
                                eval_metric=["auc", "logloss"],
                                tree_method="hist",
                                random_state=int(random_state),
                                n_jobs=-1,
                                scale_pos_weight=float(spw),
                                early_stopping_rounds=int(
                                    early_stopping_rounds
                                ),  # in constructor (warning fix)
                                verbosity=0,
                            )
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", ConvergenceWarning)
                                xgb.fit(
                                    X_train_res_scaled,
                                    y_train_res,
                                    eval_set=[
                                        (X_train_res_scaled, y_train_res),
                                        (X_val_scaled, y_val),
                                    ],
                                    verbose=False,
                                )
                            # Use validation logloss @ best_iteration to rank; tie-break by AUC-ROC
                            ev = xgb.evals_result()
                            try:
                                v_ll_series = ev["validation_1"]["logloss"]
                                v_auc_series = ev["validation_1"]["auc"]
                                v_ll = float(v_ll_series[xgb.best_iteration])
                                v_auc_roc = float(v_auc_series[xgb.best_iteration])
                            except Exception:
                                p_val_tmp = _predict_proba_safe(
                                    xgb, X_val_scaled
                                )[:, 1]
                                v_ll = log_loss(y_val, p_val_tmp, labels=[0, 1])
                                v_auc_roc = roc_auc_score(y_val, p_val_tmp)
                            _stamp(
                                f"  n={n_est:<4} lr={lr:<4} depth={md:<2} mcw={mcw:<3} "
                                f"sub={ss:<3} col={cs:<3}  val_logloss={v_ll:.4f}  "
                                f"val_auc_roc={v_auc_roc:.4f}  "
                                f"(best_it={getattr(xgb, 'best_iteration', None)})"
                            )
                            if (
                                best is None
                                or (v_ll < best["val_logloss"] - 1e-9)
                                or (
                                    abs(v_ll - best["val_logloss"]) < 1e-9
                                    and v_auc_roc > best["val_auc_roc"]
                                )
                            ):
                                best = {
                                    "params": {
                                        "n_estimators": int(n_est),
                                        "learning_rate": float(lr),
                                        "max_depth": int(md),
                                        "min_child_weight": float(mcw),
                                        "subsample": float(ss),
                                        "colsample_bytree": float(cs),
                                        "reg_lambda": float(reg_lambda),
                                    },
                                    "val_logloss": v_ll,
                                    "val_auc_roc": v_auc_roc,
                                    "model": xgb,
                                    "best_iteration": int(
                                        getattr(xgb, "best_iteration", n_est - 1)
                                    ),
                                }

    xgb_raw: XGBClassifier = best["model"]
    _stamp(
        f"[XGB] Best params: {best['params']}; "
        f"best_iteration={best['best_iteration']}"
    )

    # ---- 3) history curves from evals_result ----
    history: Dict[str, List[float]] = {"x": []}
    try:
        er = xgb_raw.evals_result()
        max_len = 0
        for split in er.values():
            for series in split.values():
                max_len = max(max_len, len(series))
        history["x"] = list(range(1, max_len + 1))

        def pick(split_key, metric):
            if split_key in er and metric in er[split_key]:
                return [float(v) for v in er[split_key][metric]]
            return []

        history["train_auc_roc"] = pick("validation_0", "auc")
        history["train_logloss"] = pick("validation_0", "logloss")
        history["val_auc_roc"] = pick("validation_1", "auc")
        history["val_logloss"] = pick("validation_1", "logloss")
    except Exception:
        pass

    # ---- 4) isotonic calibration on VALIDATION ----
    p_val_uncal = _predict_proba_safe(xgb_raw, X_val_scaled)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        p_val_uncal, np.asarray(y_val).astype(int)
    )
    xgb_cal = _IsotonicCalibrated(xgb_raw, iso)
    _stamp("[XGB] Isotonic calibration done via manual isotonic regression on validation.")

    # ---- 5) threshold tuning on VALIDATION ----
    p_val_cal = _predict_proba_safe(xgb_cal, X_val_scaled)[:, 1]
    best_t = _optimize_threshold(y_val, p_val_cal, metric=threshold_metric)

    # ---- 6) metrics on TRAIN / VAL / TEST ----
    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # Raw XGB
    p_tr_raw = _predict_proba_safe(xgb_raw, X_train_res_scaled)[:, 1]
    p_va_raw = p_val_uncal
    p_te_raw = (
        _predict_proba_safe(xgb_raw, X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["xgb_raw@0.50"] = _metrics(y_train_res, p_tr_raw, 0.5)
    out_metrics["val"]["xgb_raw@0.50"] = _metrics(y_val, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["xgb_raw@0.50"] = _metrics(y_test, p_te_raw, 0.5)

    # Calibrated XGB
    p_tr_cal = _predict_proba_safe(xgb_cal, X_train_res_scaled)[:, 1]
    p_te_cal = (
        _predict_proba_safe(xgb_cal, X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["xgb_cal@0.50"] = _metrics(y_train_res, p_tr_cal, 0.5)
    out_metrics["val"]["xgb_cal@0.50"] = _metrics(y_val, p_val_cal, 0.5)
    out_metrics["val"][f"xgb_cal@{best_t:.2f}"] = _metrics(
        y_val, p_val_cal, best_t
    )
    if p_te_cal is not None:
        out_metrics["test"]["xgb_cal@0.50"] = _metrics(y_test, p_te_cal, 0.5)
        out_metrics["test"][f"xgb_cal@{best_t:.2f}"] = _metrics(
            y_test, p_te_cal, best_t
        )

    # ---- 7) Validation & Test classification reports (PRINT + SAVE ONCE) ----
    val_key = f"xgb_cal@{best_t:.2f}"
    val_md = out_metrics["val"].get(val_key) or _metrics(y_val, p_val_cal, best_t)
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
            test_key = f"xgb_cal@{best_t:.2f}"
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

    # ---- 8) Features table (ALL or Top-K by gain importance) ----
    try:
        feat_df = _xgb_feature_table(xgb_raw, feature_names, topn_features)
    except Exception:
        feat_df = pd.DataFrame(columns=["feature", "importance"])

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
        f"XGB Importances (top {used_k})" if used_k != "all" else "XGB Importances (all)"
    )
    feat_fig = plot_features_bar(feat_df, feat_title)

    # ---- 9) Plots ----
    loss_fig = plot_xgb_loss_curves(history)
    roc_fig, pr_fig = plot_xgb_roc_pr(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="XGBoost (Calibrated)",
    )
    calib_fig = plot_xgb_calibration(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="XGBoost (Calibrated)",
    )

    # ---- 10) Build concise summaries (as DataFrames for CSV) ----
    val_summary_df = _summary_df(val_md, best_t)

    test_summary_df = pd.DataFrame()
    if have_test:
        if test_md is None and p_te_cal is not None:
            test_md = _metrics(y_test, p_te_cal, best_t)
        if test_md is not None:
            test_summary_df = _summary_df(test_md, best_t)
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

    brier_mcc_fig = plot_xgb_brier_mcc(
        val_md,
        test_md if (have_test and test_md is not None) else None,
        title_suffix="XGBoost (Calibrated)",
    )

    # ---- 11) SAVE ALL OUTPUTS ----
    if outdir is not None:
        # Full metrics tables (contain f1 and macro_f1)
        if out_metrics["train"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["train"]),
                outdir / "xgb_metrics_train",
            )
        if out_metrics["val"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["val"]),
                outdir / "xgb_metrics_val",
            )
        if out_metrics["test"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["test"]),
                outdir / "xgb_metrics_test",
            )

        # Brier & MCC summary CSV
        _save_csv(brier_mcc_df, outdir / "xgb_brier_mcc_summary")

        # Feature CSV + bar
        feat_suffix = f"top{used_k}" if used_k != "all" else "all"
        _save_csv(feat_df, outdir / f"xgb_features_{feat_suffix}")
        if feat_fig is not None:
            _save_fig(feat_fig, outdir / f"xgb_features_bar_{feat_suffix}")

        # Reports
        _save_text(val_report_text, outdir / "xgb_val_report")
        if have_test:
            _save_text(test_report_text, outdir / "xgb_test_report")

        # Summaries (CSV) — macro_avg_f1 column
        _save_csv(val_summary_df, outdir / "xgb_val_summary")
        if have_test:
            _save_csv(test_summary_df, outdir / "xgb_test_summary")

        # Run summary
        _save_text(
            "best_params: "
            + ", ".join(f"{k}={v}" for k, v in best["params"].items())
            + f"\nbest_iteration={best['best_iteration']}\n"
            f"best_threshold={best_t:.3f}\n",
            outdir / "xgb_summary",
        )

        # Plots
        _save_fig(roc_fig, outdir / "xgb_roc")
        _save_fig(pr_fig, outdir / "xgb_pr")
        _save_fig(calib_fig, outdir / "xgb_calibration")
        if loss_fig is not None:
            _save_fig(loss_fig, outdir / "xgb_loss_curve")
        if brier_mcc_fig is not None:
            _save_fig(brier_mcc_fig, outdir / "xgb_brier_mcc")

    # ---- 12) Return everything ----
    return {
        "xgb_raw": xgb_raw,
        "xgb_cal": xgb_cal,
        "best_params": best["params"],
        "best_iteration": best["best_iteration"],
        "best_threshold": best_t,
        "metrics": out_metrics,
        "history": history,
        "features": feat_df,
        "val_report_text": val_report_text,
        "test_report_text": test_report_text,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
        "brier_mcc_df": brier_mcc_df,
        "used_topn_features": used_k,
        "outdir": str(outdir) if outdir is not None else None,
    }
