__all__ = [
    "plot_all",
    "save_fig",
    "save_report_csv",
    "resolve_plot_save_dir",
    "compute_val_test_loss_curves",
    "plot_val_test_loss_curves",
]

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, log_loss
)

# ───────────────────────────────
# Save helpers / directory resolver
# ───────────────────────────────
def _ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_fig(fig, path, *, dpi: int = 150, verbose: int = 0) -> Path:
    p = _ensure_dir(Path(path).with_suffix(".png"))
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    if verbose > 0:
        print(f"✓ Saved figure: {p.resolve()}")
    return p

def save_report_csv(df: pd.DataFrame, path, *, verbose: int = 0) -> Path:
    p = _ensure_dir(Path(path).with_suffix(".csv"))
    df.to_csv(p, index=True)
    if verbose > 0:
        r, c = df.shape
        print(f"✓ Saved report: {p.resolve()}  [{r}×{c}]")
    return p

def resolve_plot_save_dir(
    *, CONFIGS: dict | None = None,
    save_dir: str | Path | None = None,
    key: str = "DIR_tr_va_te_plot_SAVE_DIR",
    verbose: int = 0
) -> Path | None:
    chosen = str(save_dir) if save_dir not in (None, "", False) else (
        str(CONFIGS.get(key)) if (CONFIGS and CONFIGS.get(key) not in (None, "", False)) else None
    )
    if not chosen:
        if verbose > 0:
            print("ℹ No save directory provided; will only display plots.")
        return None
    p = Path(chosen)
    p.mkdir(parents=True, exist_ok=True)
    if verbose > 0:
        print(f"✓ Using save directory: {p.resolve()}")
    return p

# ───────────────────────────────
# Internals
# ───────────────────────────────
def _proba(model, X):
    """Predict positive-class probability with special handling for TabNet."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        if isinstance(model, TabNetClassifier):
            Xv = X.values if hasattr(X, "values") else X
            Xv = np.asarray(Xv, dtype=np.float32)  # ensure dtype for TabNet
            return model.predict_proba(Xv)[:, 1]
    except Exception:
        pass
    return model.predict_proba(X)[:, 1]

def _choose_threshold_by_f1(y_true, p):
    ts = np.linspace(0, 1, 1001)
    f1s = [f1_score(y_true, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])

def _pick_X(result, key, split="val"):
    scaled = (key in ("lr", "mlp"))
    if split == "val":
        return result["X_val_scaled"] if scaled else result["X_val"]
    else:
        return result["X_test_scaled"] if scaled else result["X_test"]

# ───────────────────────────────
# Main “plot everything for a model” helper
# ───────────────────────────────
def plot_all(
    models, result, *,
    model_key="lr", threshold="auto",
    CONFIGS: dict | None = None,
    save_dir: str | Path | None = None,
    save_prefix: str | None = None,
    verbose: int = 0
):
    out_dir = resolve_plot_save_dir(CONFIGS=CONFIGS, save_dir=save_dir, key="DIR_tr_va_te_plot_SAVE_DIR", verbose=verbose)
    prefix = (save_prefix or model_key).strip()

    # data
    Xv = _pick_X(result, model_key, "val")
    Xt = _pick_X(result, model_key, "test")
    yv = result["y_val"].values if hasattr(result["y_val"], "values") else result["y_val"]
    yt = result["y_test"].values if hasattr(result["y_test"], "values") else result["y_test"]
    model = models[model_key]

    pv = _proba(model, Xv)
    pt = _proba(model, Xt)
    t = _choose_threshold_by_f1(yv, pv) if str(threshold).lower() == "auto" else float(threshold)

    # Optional: TabNet training history figure (loss + val AUC when available)
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        if model_key == "tabnet" and isinstance(model, TabNetClassifier) and hasattr(model, "history"):
            hist = model.history
            epochs = list(range(len(hist.get("loss", []))))
            fig = plt.figure()
            if "loss" in hist:
                plt.plot(epochs, hist["loss"], label="train_loss")
            val_auc_key = next((k for k in hist.keys() if "val_0_auc" in k or "valid_0_auc" in k), None)
            if val_auc_key is not None:
                ax1 = plt.gca(); ax2 = ax1.twinx()
                ax2.plot(epochs, hist[val_auc_key], label="val_auc")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss"); ax2.set_ylabel("Validation AUC")
                ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
                plt.title("TabNet Training History (Loss & Val AUC)")
            else:
                plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("TabNet Training History (Loss)")
            if out_dir:
                save_fig(fig, out_dir / f"{prefix}_tabnet_history", verbose=verbose)
            plt.show()
    except Exception as e:
        if verbose > 0:
            print(f"[WARN] Skipping training-history plot: {e}")

    # ROC
    fpr_v, tpr_v, _ = roc_curve(yv, pv)
    fpr_t, tpr_t, _ = roc_curve(yt, pt)
    auc_v = roc_auc_score(yv, pv); auc_t = roc_auc_score(yt, pt)
    fig = plt.figure()
    plt.plot(fpr_v, tpr_v, label=f"Val ROC (AUC={auc_v:.3f})")
    plt.plot(fpr_t, tpr_t, label=f"Test ROC (AUC={auc_t:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {model_key.upper()} (Val & Test)")
    plt.legend()
    if out_dir:
        save_fig(fig, out_dir / f"{prefix}_roc", verbose=verbose)
    plt.show()

    # PR
    prec_v, rec_v, _ = precision_recall_curve(yv, pv)
    prec_t, rec_t, _ = precision_recall_curve(yt, pt)
    ap_v = average_precision_score(yv, pv); ap_t = average_precision_score(yt, pt)
    fig = plt.figure()
    plt.plot(rec_v, prec_v, label=f"Val PR (AP={ap_v:.3f})")
    plt.plot(rec_t, prec_t, label=f"Test PR (AP={ap_t:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {model_key.upper()} (Val & Test)")
    plt.legend()
    if out_dir:
        save_fig(fig, out_dir / f"{prefix}_pr", verbose=verbose)
    plt.show()

    # Confusion matrices
    yhat_v = (pv >= t).astype(int); cm_v = confusion_matrix(yv, yhat_v)
    fig = plt.figure()
    plt.imshow(cm_v, interpolation="nearest"); plt.title(f"Confusion Matrix (Validation) — t={t:.3f}")
    plt.colorbar(); ticks = np.arange(2)
    plt.xticks(ticks, [0, 1]); plt.yticks(ticks, [0, 1])
    for i in range(cm_v.shape[0]):
        for j in range(cm_v.shape[1]):
            plt.text(j, i, str(cm_v[i, j]), ha="center", va="center")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    if out_dir:
        save_fig(fig, out_dir / f"{prefix}_cm_val", verbose=verbose)
    plt.show()

    yhat_t = (pt >= t).astype(int); cm_t = confusion_matrix(yt, yhat_t)
    fig = plt.figure()
    plt.imshow(cm_t, interpolation="nearest"); plt.title(f"Confusion Matrix (Test) — t={t:.3f}")
    plt.colorbar(); plt.xticks(ticks, [0, 1]); plt.yticks(ticks, [0, 1])
    for i in range(cm_t.shape[0]):
        for j in range(cm_t.shape[1]):
            plt.text(j, i, str(cm_t[i, j]), ha="center", va="center")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    if out_dir:
        save_fig(fig, out_dir / f"{prefix}_cm_test", verbose=verbose)
    plt.show()

    # Reports
    rep_v = classification_report(yv, yhat_v, digits=3, output_dict=True, zero_division=0); rep_v.pop("accuracy", None)
    rep_v_df = pd.DataFrame(rep_v).T
    print("\n=== Classification Report — Validation ==="); print(rep_v_df)
    if out_dir:
        save_report_csv(rep_v_df, out_dir / f"{prefix}_report_val", verbose=verbose)

    rep_t = classification_report(yt, yhat_t, digits=3, output_dict=True, zero_division=0); rep_t.pop("accuracy", None)
    rep_t_df = pd.DataFrame(rep_t).T
    print("\n=== Classification Report — Test ==="); print(rep_t_df)
    if out_dir:
        save_report_csv(rep_t_df, out_dir / f"{prefix}_report_test", verbose=verbose)

    return {
        "threshold": t,
        "auc_val": auc_v, "auc_test": auc_t,
        "ap_val": ap_v, "ap_test": ap_t,
        "cm_val": cm_v, "cm_test": cm_t,
        "report_val": rep_v_df, "report_test": rep_t_df,
    }

# ───────────────────────────────
# Logloss curves (XGB staged; TabNet per-epoch val)
# ───────────────────────────────
def _logloss_safe(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(log_loss(y, p))

def _staged_xgb_losses(model, Xv, yv, Xt, yt) -> Tuple[List[float], List[float]]:
    val_losses, test_losses = [], []
    try:
        booster = model.get_booster()
        try:
            n_rounds = booster.num_boosted_rounds()
        except Exception:
            n_rounds = len(booster.get_dump())
        n_rounds = int(max(1, min(n_rounds, 2048)))

        for i in range(1, n_rounds + 1):
            try:
                pv = model.predict_proba(Xv, iteration_range=(0, i))[:, 1]
                pt = model.predict_proba(Xt, iteration_range=(0, i))[:, 1]
            except Exception:
                pv = model.predict_proba(Xv, ntree_limit=i)[:, 1]
                pt = model.predict_proba(Xt, ntree_limit=i)[:, 1]
            val_losses.append(_logloss_safe(yv, pv))
            test_losses.append(_logloss_safe(yt, pt))
        return val_losses, test_losses
    except Exception:
        return [], []

def _tabnet_val_curve_from_history(model) -> List[float]:
    """Extract per-epoch validation logloss from TabNet history (various key names)."""
    try:
        hist = getattr(model, "history", None)
        if not hist:
            return []
        for key in ("val_0_logloss", "valid_0_logloss", "val_logloss", "valid_logloss"):
            if key in hist and isinstance(hist[key], (list, tuple)):
                return list(hist[key])
        # some versions nest history
        if hasattr(hist, "history"):
            h = hist.history
            for key in ("val_0_logloss", "valid_0_logloss", "val_logloss", "valid_logloss"):
                if key in h:
                    return list(h[key])
    except Exception:
        pass
    return []

def compute_val_test_loss_curves(models, result, *, model_key: str) -> Dict[str, List[float]]:
    model = models[model_key]
    mk = model_key.lower()

    # XGB staged curves
    try:
        from xgboost import XGBClassifier  # noqa: F401
        if hasattr(model, "get_booster") and mk == "xgb":
            Xv = _pick_X(result, model_key, "val")
            Xt = _pick_X(result, model_key, "test")
            yv = result["y_val"].values if hasattr(result["y_val"], "values") else result["y_val"]
            yt = result["y_test"].values if hasattr(result["y_test"], "values") else result["y_test"]
            v_curve, t_curve = _staged_xgb_losses(model, Xv, yv, Xt, yt)
            if v_curve or t_curve:
                return {"val": v_curve, "test": t_curve}
    except Exception:
        pass

    # TabNet per-epoch validation curve
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
        if mk == "tabnet":
            val_curve = _tabnet_val_curve_from_history(model)
            if val_curve:
                return {"val": val_curve, "test": []}
    except Exception:
        pass

    # Others: nothing available
    return {"val": [], "test": []}

def plot_val_test_loss_curves(
    models, result, *,
    model_key: str = "xgb",
    CONFIGS: dict | None = None,
    save_dir: str | Path | None = None,
    save_prefix: str | None = None,
    verbose: int = 0,
):
    out_dir = resolve_plot_save_dir(CONFIGS=CONFIGS, save_dir=save_dir, key="DIR_tr_va_te_plot_SAVE_DIR", verbose=verbose)
    prefix = (save_prefix or model_key).strip()
    mk = model_key.lower()
    model = models[model_key]

    # final reference losses
    Xv = _pick_X(result, model_key, "val")
    Xt = _pick_X(result, model_key, "test")
    yv = result["y_val"].values if hasattr(result["y_val"], "values") else result["y_val"]
    yt = result["y_test"].values if hasattr(result["y_test"], "values") else result["y_test"]

    try:
        from utils.helpers.tr_va_te_eval_helpers import proba as _safe_proba  # optional
        pv = _safe_proba(model, Xv); pt = _safe_proba(model, Xt)
    except Exception:
        pv = _proba(model, Xv);      pt = _proba(model, Xt)

    val_ll = _logloss_safe(yv, pv)
    test_ll = _logloss_safe(yt, pt)

    # curves
    curves = compute_val_test_loss_curves(models, result, model_key=model_key)
    v_curve, t_curve = curves.get("val", []), curves.get("test", [])

    fig = plt.figure()
    ax = plt.gca()

    if v_curve:
        x = range(1, len(v_curve) + 1)
        label = "Validation logloss (staged)" if mk == "xgb" else "Validation logloss (per-epoch)"
        ax.plot(x, v_curve, label=label)
    if t_curve:
        ax.plot(range(1, len(t_curve) + 1), t_curve, label="Test logloss (staged)")

    ax.axhline(val_ll, linestyle="--", label=f"Final Val logloss = {val_ll:.4f}")
    ax.axhline(test_ll, linestyle=":",  label=f"Final Test logloss = {test_ll:.4f}")

    ax.set_xlabel("Iteration / Trees" if mk == "xgb" else "Epoch")
    ax.set_ylabel("Log Loss")
    title_suffix = (
        " (staged)" if mk == "xgb" and (v_curve or t_curve)
        else " (per-epoch val)" if mk == "tabnet" and v_curve
        else " (final refs only)"
    )
    ax.set_title(f"{model_key.upper()} — Validation/Test Logloss{title_suffix}")
    ax.legend()

    if out_dir:
        save_fig(fig, Path(out_dir) / f"{prefix}_val_test_logloss", verbose=verbose)
    plt.show()

    if verbose > 0 and not (v_curve or t_curve):
        if mk == "xgb":
            print("ℹ XGBoost: no staged curves available; showing final logloss lines.")
        elif mk == "tabnet":
            print("ℹ TabNet: no per-epoch val logloss found in model.history; showing final logloss lines.")
        else:
            print("ℹ This model does not expose per-iteration validation/test curves; showing final logloss lines.")
