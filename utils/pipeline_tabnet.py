# ============================
# TabNet Pipeline (train/validate/test) with saving — corrected
# - Extended metrics: auc_roc, auc_pr, logloss, brier, acc, F1 (pos), macro_F1, MCC
# - In *_summary.csv: F1 column renamed to macro_avg_f1 (macro-averaged F1)
# - AUC renamed to auc_roc; added auc_pr
# - Manual isotonic calibration on validation (no cv='prefit')
# - Threshold tuning (F1 or Youden)
# - Saves: metrics CSVs, val/test reports, val/test summary CSVs,
#          ROC/PR/Calibration/Loss plots, feature importance CSV + bar plot,
#          Brier+MCC CSV + bar plot
# ============================
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.exceptions import ConvergenceWarning

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAVE_TABNET = True
except Exception:
    HAVE_TABNET = False

# Torch (for cuda checks / memory management niceties)
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


# ---------- basic io helpers ----------
def _stamp(msg: str):
    print(msg)


def _ensure_dir(p: Path) -> Path:
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


def _outdir(CONFIGS):
    p = Path(CONFIGS["DIR_tr_va_te_metric_shap_SAVE_DIR"]) / "tabnet"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- utils ----------
class _IsotonicCalibrated:
    """Wrap a prefit classifier (with predict_proba) using isotonic calib trained on validation probs."""

    def __init__(self, base_estimator, iso: IsotonicRegression):
        self.base_estimator = base_estimator
        self.iso = iso
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        p = self.base_estimator.predict_proba(_to_numpy(X))[:, 1]
        pc = np.clip(self.iso.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _to_numpy(X) -> np.ndarray:
    """Force numpy float32 arrays for TabNet to avoid pandas __getitem__ KeyError in PredictDataset."""
    if hasattr(X, "values"):
        a = X.values
    else:
        a = np.asarray(X)
    a = a.astype(np.float32, copy=False)
    a[~np.isfinite(a)] = 0.0
    return a


def _balanced_sample_weight(y) -> np.ndarray:
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=np.float32)
    w_pos = 0.5 / pos
    w_neg = 0.5 / neg
    sw = np.where(y == 1, w_pos, w_neg)
    return (sw * y.size).astype(np.float32)


def _metrics(y_true, p, t: float = 0.5) -> Dict[str, float]:
    """
    Extended metrics for model comparison.

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


def _optimize_threshold(y_true, p, *, metric: str = "f1") -> float:
    """
    Threshold tuning, typically on validation probabilities.

    metric="f1" uses positive-class F1; metric="youden" uses Youden's J.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)
    grid = np.linspace(0.05, 0.95, 19)

    if str(metric).lower() == "youden":
        from sklearn.metrics import recall_score

        def scorer(t):
            yhat = (p >= t).astype(int)
            tn = np.sum((y_true == 0) & (yhat == 0))
            fp = np.sum((y_true == 0) & (yhat == 1))
            spec = tn / (tn + fp + 1e-9)
            sens = recall_score(y_true, yhat)
            return sens + spec - 1.0

    else:  # F1 (positive class)
        def scorer(t):
            return f1_score(y_true, (p >= t).astype(int))

    best_t, best_s = 0.5, -1.0
    for t in grid:
        s = scorer(float(t))
        if s > best_s:
            best_s, best_t = s, float(t)
    _stamp(f"[TABNET][THRESH] best_t={best_t:.3f} ({str(metric).lower()}={best_s:.4f})")
    return float(best_t)


def _metrics_split_to_df(split_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, md in split_dict.items():
        row = {"model": name}
        row.update(md)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- plots ----------
def plot_tabnet_roc_pr(
    y_val, p_val, y_test, p_test, title_suffix: str = "TabNet (calibrated)"
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


def plot_tabnet_calibration(
    y_val,
    p_val,
    y_test=None,
    p_test=None,
    n_bins: int = 15,
    title_suffix: str = "TabNet (calibrated)",
):
    from sklearn.calibration import calibration_curve

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


def plot_loss_curves(history: Dict[str, List[float]]):
    if not history:
        return None
    xs = history.get("epoch", [])
    tr = history.get("train_loss", [])
    va = history.get("val_logloss", [])
    if not xs or (not tr and not va):
        return None
    fig = plt.figure()
    if tr:
        plt.plot(xs[: len(tr)], tr, label="train loss")
    if va:
        plt.plot(xs[: len(va)], va, label="val logloss")
    plt.title("TabNet: loss curves")
    plt.xlabel("epoch")
    plt.ylabel("loss/logloss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_feature_bar(
    df_imp: pd.DataFrame, title: str = "TabNet Features (|importance|)"
):
    if df_imp is None or df_imp.empty:
        return None
    fig = plt.figure(figsize=(7, max(5, 0.30 * len(df_imp))))
    y = np.arange(len(df_imp))
    vals = df_imp["importance"].values
    plt.barh(y, vals)
    plt.yticks(y, df_imp["feature"].values)
    plt.gca().invert_yaxis()  # biggest at top
    offset = 0.01 * (vals.max() if len(vals) else 1.0)
    for i, v in enumerate(vals):
        plt.text(v + offset, i, f"{v:.3f}", va="center", ha="left", fontsize=8)
    plt.xlabel("importance")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_tabnet_brier_mcc(
    val_md: Dict[str, float],
    test_md: Optional[Dict[str, float]] = None,
    title_suffix: str = "TabNet (Calibrated)",
):
    """Bar plot for Brier and MCC on validation (and test if provided)."""
    metrics = ["brier", "mcc"]
    labels = [m.upper() for m in metrics]
    val_vals = [float(val_md.get(m, np.nan)) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig = plt.figure()
    ax = plt.gca()

    if test_md is not None:
        ax.bar(x - width / 2, val_vals, width, label="Validation")
        test_vals = [float(test_md.get(m, np.nan)) for m in metrics]
        ax.bar(x + width / 2, test_vals, width, label="Test")
    else:
        ax.bar(x, val_vals, width * 1.2, label="Validation")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Brier & MCC — {title_suffix}")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y")
    ax.legend()

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


# ---------- single trial fit ----------
def _fit_tabnet_once(
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    n_d: int,
    n_steps: int,
    gamma: float,
    n_shared: int,
    n_independent: int,
    lambda_sparse: float,
    mask_type: str,
    device_name: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    virtual_batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    cat_idxs: List[int],
    cat_dims: List[int],
    cat_emb_dim: List[int],
    sample_weights: Optional[np.ndarray],
) -> Tuple[Any, float, float, np.ndarray, Dict[str, List[float]]]:
    """Returns model, val_logloss, val_auc, p_val, history."""
    assert HAVE_TABNET, "pytorch_tabnet not installed."

    vbs = int(min(max(16, virtual_batch_size), max(16, batch_size)))
    if vbs > batch_size:
        vbs = max(16, batch_size // 2) if batch_size >= 32 else 16

    try:
        tn = TabNetClassifier(
            n_d=n_d,
            n_a=n_d,
            n_steps=n_steps,
            gamma=gamma,
            n_shared=n_shared,
            n_independent=n_independent,
            lambda_sparse=lambda_sparse,
            mask_type=mask_type,
            seed=seed,
            verbose=0,
            device_name=device_name,
            cat_idxs=list(cat_idxs or []),
            cat_dims=list(cat_dims or []),
            cat_emb_dim=list(cat_emb_dim or []),
        )
    except TypeError:
        # Last-resort safety: force empty categorical setup
        tn = TabNetClassifier(
            n_d=n_d,
            n_a=n_d,
            n_steps=n_steps,
            gamma=gamma,
            n_shared=n_shared,
            n_independent=n_independent,
            lambda_sparse=lambda_sparse,
            mask_type=mask_type,
            seed=seed,
            verbose=0,
            device_name=device_name,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=[],
        )

    tn.fit(
        Xtr,
        np.asarray(ytr).astype(int),
        eval_set=[(Xva, np.asarray(yva).astype(int))],
        eval_name=["val"],
        eval_metric=["auc", "logloss"],
        max_epochs=int(max_epochs),
        patience=int(patience),
        batch_size=int(batch_size),
        virtual_batch_size=int(vbs),
        num_workers=0,
        drop_last=False,
        weights=sample_weights,
        pin_memory=(device_name == "cuda"),
    )

    p_val = tn.predict_proba(Xva)[:, 1]
    v_ll = log_loss(yva, p_val, labels=[0, 1])
    v_auc = roc_auc_score(yva, p_val)

    # history extraction (robust)
    hist = {"epoch": [], "train_loss": [], "val_logloss": [], "val_auc": []}
    try:
        if hasattr(tn, "history") and tn.history:
            H = tn.history if isinstance(tn.history, dict) else getattr(
                tn.history, "history", {}
            )
            if isinstance(H, dict):
                if "loss" in H:
                    hist["train_loss"] = list(H["loss"])
                if "val_0_logloss" in H:
                    hist["val_logloss"] = list(H["val_0_logloss"])
                elif "valid_0_logloss" in H:
                    hist["val_logloss"] = list(H["valid_0_logloss"])
                if "val_0_auc" in H:
                    hist["val_auc"] = list(H["val_0_auc"])
                elif "valid_0_auc" in H:
                    hist["val_auc"] = list(H["valid_0_auc"])
                mlen = max(
                    len(hist["train_loss"]),
                    len(hist["val_logloss"]),
                    len(hist["val_auc"]),
                )
                hist["epoch"] = list(range(1, mlen + 1))
    except Exception:
        pass

    return tn, float(v_ll), float(v_auc), p_val, hist


# ---------- main pipeline ----------
def train_validate_test_tabnet(
    X_train_res_scaled,
    y_train_res,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test,
    feature_names: Optional[List[str]] = None,
    *,
    width: int = 24,
    n_steps: int = 4,
    n_shared: int = 2,
    n_independent: int = 2,
    gamma: float = 1.5,
    lambda_sparse: float = 1e-5,
    mask_type: str = "sparsemax",
    lr: float = 6e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 120,
    patience: int = 40,
    batch_size: int = 256,
    virtual_batch_size: int = 32,
    random_state: int = 42,
    cat_idxs: Optional[List[int]] = None,
    cat_dims: Optional[List[int]] = None,
    cat_emb_dim: Optional[int | List[int]] = 8,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,  # 0/None => all
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True,
    device: Optional[str] = None,  # 'cuda' or 'cpu' or None (auto)
    max_trials: int = 5,  # limit scan size
    early_stop_no_improve: int = 2,  # require improvements across trials
) -> Dict[str, Any]:
    """
    Full TabNet pipeline on *scaled* splits + exports.

    Summary CSVs (tabnet_val_summary.csv / tabnet_test_summary.csv) contain:
      - threshold
      - auc_roc
      - auc_pr
      - logloss
      - brier
      - acc
      - macro_avg_f1   (macro-averaged F1; replaces 'f1' in summary)
      - mcc
    """
    if not HAVE_TABNET:
        raise ImportError("pytorch_tabnet is not installed.")

    outdir = _outdir(CONFIGS) if (save_outputs and CONFIGS) else None
    if outdir is not None:
        print(f"[SAVE] Output directory: {outdir.resolve()}")

    # ---- feature names ----
    if feature_names is None:
        feature_names = list(
            getattr(
                X_train_res_scaled,
                "columns",
                [f"f{i}" for i in range(_to_numpy(X_train_res_scaled).shape[1])],
            )
        )

    # ---- device preference ----
    cuda_ok = HAVE_TORCH and torch.cuda.is_available()
    pref = (
        device
        or (CONFIGS.get("DEVICE") if CONFIGS else None)
        or ("cuda" if cuda_ok else "cpu")
    )
    _stamp(f"[TABNET] Device preference: {pref} (cuda_available={cuda_ok})")

    # ---- class balancing ----
    sw = _balanced_sample_weight(y_train_res)

    # ---- small grid (n_d, n_steps, batch_size) ----
    n_d_list = sorted({max(8, int(width)), max(8, int(width // 2))})
    steps_list = sorted({max(3, int(n_steps)), max(3, int(n_steps) - 1)})
    bs_list = sorted({max(32, int(batch_size)), max(32, int(batch_size // 2))})
    grid: List[Tuple[int, int, int]] = [
        (nd, st, bs) for nd in n_d_list for st in steps_list for bs in bs_list
    ]
    grid = grid[: max_trials]

    # ---- categorical meta: always lists ----
    cidx = list(cat_idxs) if cat_idxs is not None else []
    cdims = list(cat_dims) if cat_dims is not None else []

    if isinstance(cat_emb_dim, int):
        cemb = [int(cat_emb_dim)] * len(cdims)
    elif isinstance(cat_emb_dim, (list, tuple, np.ndarray)):
        cemb = list(cat_emb_dim)
    else:
        cemb = []

    if len(cemb) != len(cdims):
        cemb = ([8] * len(cdims)) if len(cdims) > 0 else []

    # ---- arrays for TabNet ----
    Xtr, Xva = _to_numpy(X_train_res_scaled), _to_numpy(X_val_scaled)
    Xte = _to_numpy(X_test_scaled) if X_test_scaled is not None else None
    ytr, yva = np.asarray(y_train_res).astype(int), np.asarray(y_val).astype(int)
    yte = np.asarray(y_test).astype(int) if y_test is not None else None

    best_model = None
    best_tuple = None
    best_ll, best_auc = np.inf, -np.inf
    history_best: Dict[str, List[float]] = {}

    _stamp("[TABNET] Hyperparam scan (n_d/steps/batch_size):")
    trials_no_improve = 0

    device_order = [pref] if pref == "cpu" else [pref, "cpu"]
    for (n_d_try, steps_try, bs_try) in grid:
        trial_improved = False
        for dev in device_order:
            try:
                mdl, v_ll, v_auc, p_val, hist = _fit_tabnet_once(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    n_d=n_d_try,
                    n_steps=steps_try,
                    gamma=gamma,
                    n_shared=n_shared,
                    n_independent=n_independent,
                    lambda_sparse=lambda_sparse,
                    mask_type=mask_type,
                    device_name=("cuda" if (dev == "cuda" and cuda_ok) else "cpu"),
                    lr=lr,
                    weight_decay=weight_decay,
                    batch_size=bs_try,
                    virtual_batch_size=virtual_batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=random_state,
                    cat_idxs=cidx,
                    cat_dims=cdims,
                    cat_emb_dim=cemb,
                    sample_weights=sw,
                )
                _stamp(
                    f"  n_d={n_d_try:<3}  steps={steps_try:<1}  bs={bs_try:<3}  "
                    f"dev={('cuda' if (dev == 'cuda' and cuda_ok) else 'cpu'):<4}  "
                    f"val_logloss={v_ll:.4f}  val_auc={v_auc:.4f}"
                )
                if (v_ll < best_ll - 1e-9) or (
                    abs(v_ll - best_ll) < 1e-9 and v_auc > best_auc
                ):
                    best_ll, best_auc = v_ll, v_auc
                    best_model, best_tuple = mdl, (n_d_try, steps_try, bs_try)
                    history_best = hist
                    trial_improved = True
                break  # success for this (n_d,steps,bs)
            except RuntimeError as e:
                msg = str(e).lower()
                if (dev == "cuda") and ("out of memory" in msg or "cuda" in msg):
                    _stamp(
                        f"[TABNET] CUDA trial failed ({msg[:80]}...) → falling back to CPU"
                    )
                    try:
                        if HAVE_TORCH:
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue  # try on CPU
                else:
                    raise
            except TypeError as e:
                _stamp(
                    f"[TABNET] Embedding param error: {e}. Retrying with no categorical metadata."
                )
                mdl, v_ll, v_auc, p_val, hist = _fit_tabnet_once(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    n_d=n_d_try,
                    n_steps=steps_try,
                    gamma=gamma,
                    n_shared=n_shared,
                    n_independent=n_independent,
                    lambda_sparse=lambda_sparse,
                    mask_type=mask_type,
                    device_name=("cuda" if (dev == "cuda" and cuda_ok) else "cpu"),
                    lr=lr,
                    weight_decay=weight_decay,
                    batch_size=bs_try,
                    virtual_batch_size=virtual_batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=random_state,
                    cat_idxs=[],
                    cat_dims=[],
                    cat_emb_dim=[],
                    sample_weights=sw,
                )
                _stamp(
                    f"  n_d={n_d_try:<3}  steps={steps_try:<1}  bs={bs_try:<3}  "
                    f"dev={('cuda' if (dev == 'cuda' and cuda_ok) else 'cpu'):<4}  "
                    f"val_logloss={v_ll:.4f}  val_auc={v_auc:.4f}"
                )
                if (v_ll < best_ll - 1e-9) or (
                    abs(v_ll - best_ll) < 1e-9 and v_auc > best_auc
                ):
                    best_ll, best_auc = v_ll, v_auc
                    best_model, best_tuple = mdl, (n_d_try, steps_try, bs_try)
                    history_best = hist
                    trial_improved = True
                break

        if not trial_improved:
            trials_no_improve += 1
            if trials_no_improve >= max(1, int(early_stop_no_improve)):
                # early stopping over outer hyperparam trials (optional)
                pass

    if best_model is None:
        raise RuntimeError("TabNet fit failed for all scanned configurations.")
    n_d_best, steps_best, bs_best = best_tuple
    _stamp(
        f"[TABNET] Best: n_d={n_d_best}, steps={steps_best}, batch_size={bs_best}"
    )

    # ---- isotonic calibration (on validation) ----
    p_val_uncal = best_model.predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_uncal, yva)
    cal_model = _IsotonicCalibrated(best_model, iso)
    _stamp("[TABNET] Isotonic calibration done on validation.")

    # ---- threshold selection on validation ----
    p_val_cal = cal_model.predict_proba(Xva)[:, 1]
    best_t = _optimize_threshold(yva, p_val_cal, metric=threshold_metric)

    # ---- metrics ----
    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # Raw (uncalibrated)
    p_tr_raw = best_model.predict_proba(Xtr)[:, 1]
    p_va_raw = p_val_uncal
    p_te_raw = best_model.predict_proba(Xte)[:, 1] if Xte is not None else None
    out_metrics["train"]["tabnet_raw@0.50"] = _metrics(ytr, p_tr_raw, 0.5)
    out_metrics["val"]["tabnet_raw@0.50"] = _metrics(yva, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["tabnet_raw@0.50"] = _metrics(yte, p_te_raw, 0.5)

    # Calibrated
    p_tr_cal = cal_model.predict_proba(Xtr)[:, 1]
    p_te_cal = cal_model.predict_proba(Xte)[:, 1] if Xte is not None else None
    out_metrics["train"]["tabnet_cal@0.50"] = _metrics(ytr, p_tr_cal, 0.5)
    out_metrics["val"]["tabnet_cal@0.50"] = _metrics(yva, p_val_cal, 0.5)
    out_metrics["val"][f"tabnet_cal@{best_t:.2f}"] = _metrics(
        yva, p_val_cal, best_t
    )
    if p_te_cal is not None:
        out_metrics["test"]["tabnet_cal@0.50"] = _metrics(yte, p_te_cal, 0.5)
        out_metrics["test"][f"tabnet_cal@{best_t:.2f}"] = _metrics(
            yte, p_te_cal, best_t
        )

    # ---- summary helpers (for *_summary.csv) ----
    def _summary_df(md: Dict[str, float], threshold: float) -> pd.DataFrame:
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
            "macro_avg_f1": md.get("macro_f1", md.get("f1", np.nan)),
            "mcc": md.get("mcc", np.nan),
        }
        return pd.DataFrame([row], columns=cols)

    # ---- derive val/test summaries and Brier/MCC summary ----
    val_key = f"tabnet_cal@{best_t:.2f}"
    val_md = out_metrics["val"].get(val_key) or _metrics(yva, p_val_cal, best_t)
    val_summary_df = _summary_df(val_md, best_t)

    test_summary_df = pd.DataFrame()
    test_md: Optional[Dict[str, float]] = None
    if Xte is not None and p_te_cal is not None:
        test_key = f"tabnet_cal@{best_t:.2f}"
        test_md = out_metrics["test"].get(test_key) or _metrics(
            yte, p_te_cal, best_t
        )
        test_summary_df = _summary_df(test_md, best_t)

    brier_mcc_rows = [
        {
            "split": "val",
            "threshold": best_t,
            "brier": float(val_md.get("brier", np.nan)),
            "mcc": float(val_md.get("mcc", np.nan)),
        }
    ]
    if Xte is not None and test_md is not None:
        brier_mcc_rows.append(
            {
                "split": "test",
                "threshold": best_t,
                "brier": float(test_md.get("brier", np.nan)),
                "mcc": float(test_md.get("mcc", np.nan)),
            }
        )
    brier_mcc_df = pd.DataFrame(brier_mcc_rows)

    # ---- reports (print ONCE each) ----
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
        f"{classification_report(yva, y_val_pred, target_names=['Class 0','Class 1'], digits=2)}"
    )
    print("\n" + val_report_text)

    test_report_text = ""
    if Xte is not None and p_te_cal is not None and test_md is not None:
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
            f"{classification_report(yte, y_pred_test, target_names=['Class 0','Class 1'], digits=2)}"
        )
        print("\n" + test_report_text)

    # ---- feature importance ----
    try:
        imp = getattr(best_model, "feature_importances_", None)
        if imp is None:
            features_df = pd.DataFrame(columns=["feature", "importance"])
        else:
            imp = np.asarray(imp).reshape(-1)
            features_df = (
                pd.DataFrame({"feature": feature_names, "importance": np.abs(imp)})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            if topn_features and topn_features > 0:
                features_df = features_df.head(int(topn_features)).reset_index(
                    drop=True
                )
    except Exception:
        features_df = pd.DataFrame(columns=["feature", "importance"])

    # ---- plots ----
    roc_fig, pr_fig = plot_tabnet_roc_pr(
        yva,
        p_val_cal,
        (yte if Xte is not None else None),
        (p_te_cal if Xte is not None else None),
        title_suffix="TabNet (Calibrated)",
    )
    calib_fig = plot_tabnet_calibration(
        yva,
        p_val_cal,
        (yte if Xte is not None else None),
        (p_te_cal if Xte is not None else None),
        title_suffix="TabNet (Calibrated)",
    )
    loss_fig = plot_loss_curves(history_best)
    feat_fig = plot_feature_bar(features_df, title="TabNet Feature Importances")
    brier_mcc_fig = plot_tabnet_brier_mcc(
        val_md, test_md if Xte is not None and test_md is not None else None
    )

    # ---- SAVE ALL OUTPUTS ----
    if outdir is not None:
        # metrics CSVs (full metrics including auc_roc, auc_pr, macro_f1, mcc)
        if out_metrics["train"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["train"]),
                outdir / "tabnet_metrics_train",
            )
        if out_metrics["val"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["val"]),
                outdir / "tabnet_metrics_val",
            )
        if out_metrics["test"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["test"]),
                outdir / "tabnet_metrics_test",
            )

        # reports
        _save_text(val_report_text, outdir / "tabnet_val_report")
        if test_report_text:
            _save_text(test_report_text, outdir / "tabnet_test_report")

        # summaries (CSV) — with macro_avg_f1 column
        _save_csv(val_summary_df, outdir / "tabnet_val_summary")
        if not test_summary_df.empty:
            _save_csv(test_summary_df, outdir / "tabnet_test_summary")

        # Brier/MCC summary
        _save_csv(brier_mcc_df, outdir / "tabnet_brier_mcc_summary")

        # features
        _save_csv(features_df, outdir / "tabnet_features")

        # plots
        _save_fig(roc_fig, outdir / "tabnet_roc")
        _save_fig(pr_fig, outdir / "tabnet_pr")
        _save_fig(calib_fig, outdir / "tabnet_calibration")
        if loss_fig is not None:
            _save_fig(loss_fig, outdir / "tabnet_loss_curve")
        if feat_fig is not None:
            _save_fig(feat_fig, outdir / "tabnet_features_bar")
        if brier_mcc_fig is not None:
            _save_fig(brier_mcc_fig, outdir / "tabnet_brier_mcc")

    # ---- return payload ----
    return {
        "tabnet_model": best_model,
        "tabnet_calibrated": cal_model,
        "best_params": {
            "n_d": n_d_best,
            "n_steps": steps_best,
            "batch_size": bs_best,
        },
        "best_threshold": best_t,
        "metrics": out_metrics,
        "history": history_best,
        "features": features_df,
        "val_report_text": val_report_text,
        "val_summary_df": val_summary_df,
        "test_report_text": test_report_text,
        "test_summary_df": test_summary_df,
        "brier_mcc_df": brier_mcc_df,
        "outdir": (str(outdir) if outdir is not None else None),
    }
