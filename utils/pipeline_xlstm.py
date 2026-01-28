# ============================
# XLSTM Pipeline (AMP fixed, OOM-safe, reports & CSVs) — full corrected code
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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -------- IO helpers --------
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
    p = Path(CONFIGS["DIR_tr_va_te_metric_shap_SAVE_DIR"]) / "xlstm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _select_device(CONFIGS: Dict[str, Any]) -> str:
    if CONFIGS.get("USE_GPU") is False:
        return "cpu"
    dev = str(CONFIGS.get("DEVICE", "")).strip().lower()
    if dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------- data & metrics helpers --------
def _to_float32_df(X) -> np.ndarray:
    a = X.values if hasattr(X, "values") else np.asarray(X)
    a = a.astype(np.float32, copy=False)
    a[~np.isfinite(a)] = 0.0
    return a


def _make_seq(X2d: np.ndarray) -> np.ndarray:
    # (N, F) -> (N, F, 1) as a pseudo-sequence across features
    return X2d.reshape(X2d.shape[0], X2d.shape[1], 1).astype(np.float32)


def _pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1)) if pos > 0 else 1.0


def _safe_log_loss(y_true, p):
    p = np.clip(np.asarray(p).reshape(-1), 1e-7, 1 - 1e-7)
    return log_loss(np.asarray(y_true).astype(int), p, labels=[0, 1])


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
    ll = _safe_log_loss(y_true, p1)
    brier = brier_score_loss(y_true, p1)
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)  # positive-class F1
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
    _stamp(f"[XLSTM][THRESH] best_t={best_t:.3f} ({metric}={best_s:.4f})")
    return float(best_t)


# -------- isotonic wrapper (calibrated predict_proba -> 2 columns) --------
class _IsotonicCalibrated:
    def __init__(self, base_model, iso: IsotonicRegression):
        self.base_model = base_model
        self.iso = iso
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        p = self.base_model.predict_proba(X)  # 1-D positive probs
        pc = np.clip(self.iso.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# -------- model --------
class XLSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # x: (B, F, 1)
        _, (hn, _) = self.lstm(x)
        h_last = hn[-1]
        return self.head(h_last).squeeze(1)


class XLSTMWrapper:
    """AMP on GPU (new torch.amp), grad clipping, OOM-safe batch/device fallback."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        batch_size=256,
        max_epochs=100,
        patience=20,
        random_state=42,
        device: Optional[str] = None,
        pos_weight: float = 1.0,
        clip_grad: float = 1.0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_weight = float(pos_weight)
        self.clip_grad = float(clip_grad)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.model: Optional[nn.Module] = None
        self.history_: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_logloss": [],
            "val_auc": [],
        }
        self._best_state = None

        self._init_scaler()
        self._build_model_oom_safe()

    def _init_scaler(self):
        # AMP enabled only on CUDA device
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))

    def _reinit_scaler_for_device(self):
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=(self.device == "cuda"))

    def _build_model_oom_safe(self):
        try:
            self.model = XLSTMNet(
                self.input_dim, self.hidden_dim, self.num_layers, self.dropout
            ).to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device == "cuda":
                _stamp("[XLSTM] CUDA OOM creating model; fallback to CPU.")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = "cpu"
                self._reinit_scaler_for_device()
                self.model = XLSTMNet(
                    self.input_dim, self.hidden_dim, self.num_layers, self.dropout
                ).to(self.device)
            else:
                raise

    @torch.no_grad()
    def _infer_proba(
        self, model: nn.Module, X2d: np.ndarray, batch_size: int
    ) -> np.ndarray:
        model.eval()
        Xseq = _make_seq(X2d)
        ds = TensorDataset(torch.from_numpy(Xseq))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        out = []
        for (xb,) in dl:
            xb = xb.to(self.device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            out.append(p)
        return np.concatenate(out, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        X2d = _to_float32_df(X)
        return self._infer_proba(self.model, X2d, self.batch_size)

    def fit(self, X_train, y_train, X_val, y_val):
        Xtr = _to_float32_df(X_train)
        Xva = _to_float32_df(X_val)
        ytr = np.asarray(y_train).astype(np.float32)
        yva = np.asarray(y_val).astype(int)

        ds_tr = TensorDataset(
            torch.from_numpy(_make_seq(Xtr)), torch.from_numpy(ytr)
        )
        ds_va = TensorDataset(
            torch.from_numpy(_make_seq(Xva)),
            torch.from_numpy(yva.astype(np.float32)),
        )

        current_bs = max(32, int(self.batch_size))
        self._build_model_oom_safe()

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.pos_weight, dtype=torch.float32, device=self.device
            )
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_ll = np.inf
        wait = 0
        epoch = 0

        while epoch < self.max_epochs:
            dl_tr = DataLoader(
                ds_tr, batch_size=current_bs, shuffle=True, drop_last=False
            )
            dl_va = DataLoader(
                ds_va, batch_size=max(1024, current_bs), shuffle=False
            )

            try:
                # ---- train ----
                self.model.train()
                tr_loss = 0.0
                nobs = 0
                for xb, yb in dl_tr:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad(set_to_none=True)

                    with torch.amp.autocast(
                        device_type="cuda", enabled=(self.device == "cuda")
                    ):
                        logits = self.model(xb)
                        loss = criterion(logits, yb)

                    self.scaler.scale(loss).backward()
                    if self.clip_grad > 0:
                        self.scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.clip_grad
                        )
                    self.scaler.step(opt)
                    self.scaler.update()

                    bs = yb.shape[0]
                    tr_loss += float(loss.item()) * bs
                    nobs += bs

                tr_loss /= max(nobs, 1)

                # ---- validate ----
                self.model.eval()
                with torch.no_grad():
                    pv = self._infer_proba(
                        self.model, Xva, max(1024, current_bs)
                    )
                val_ll = _safe_log_loss(yva, pv)
                val_auc = roc_auc_score(yva, pv)

                epoch += 1
                self.history_["epoch"].append(epoch)
                self.history_["train_loss"].append(tr_loss)
                self.history_["val_logloss"].append(val_ll)
                self.history_["val_auc"].append(val_auc)

                improved = val_ll < best_val_ll - 1e-9
                if improved:
                    best_val_ll = val_ll
                    self._best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1

                if (epoch % 5) == 0 or epoch == 1:
                    _stamp(
                        f"[XLSTM] epoch={epoch:03d} "
                        f"train_loss={tr_loss:.4f} val_logloss={val_ll:.4f} val_auc={val_auc:.4f}"
                    )

                if wait >= self.patience:
                    _stamp(
                        f"[XLSTM] Early stop at epoch={epoch} (patience={self.patience})."
                    )
                    break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.device == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if current_bs > 32:
                            new_bs = max(32, current_bs // 2)
                            _stamp(
                                f"[XLSTM] CUDA OOM; reduce batch_size {current_bs} → {new_bs} and retry epoch."
                            )
                            current_bs = new_bs
                            continue
                        else:
                            _stamp(
                                "[XLSTM] CUDA OOM at min batch; switching to CPU and retrying."
                            )
                            self.device = "cpu"
                            self._reinit_scaler_for_device()
                            self._build_model_oom_safe()
                            criterion = nn.BCEWithLogitsLoss(
                                pos_weight=torch.tensor(
                                    self.pos_weight,
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            opt = torch.optim.Adam(
                                self.model.parameters(), lr=self.lr
                            )
                            current_bs = max(64, current_bs)
                            continue
                    else:
                        raise
                else:
                    raise

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.batch_size = current_bs
        return self


# -------- plots --------
def plot_xlstm_loss_curves(hist: Dict[str, List[float]]):
    if not hist or "epoch" not in hist:
        return None
    fig = plt.figure()
    if hist.get("train_loss"):
        plt.plot(
            hist["epoch"][: len(hist["train_loss"])],
            hist["train_loss"],
            label="train loss",
        )
    if hist.get("val_logloss"):
        plt.plot(
            hist["epoch"][: len(hist["val_logloss"])],
            hist["val_logloss"],
            label="val logloss",
        )
    if hist.get("val_auc"):
        plt.plot(
            hist["epoch"][: len(hist["val_auc"])],
            hist["val_auc"],
            label="val AUC-ROC",
        )
    plt.title("XLSTM: training curves")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_xlstm_roc_pr(
    y_val,
    p_val,
    y_test,
    p_test,
    title_suffix: str = "XLSTM (calibrated)",
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


def plot_xlstm_calibration(
    y_val,
    p_val,
    y_test=None,
    p_test=None,
    n_bins: int = 15,
    title_suffix: str = "XLSTM (calibrated)",
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


def plot_features_bar(
    df_top: pd.DataFrame, title: str = "XLSTM Permutation Importance (Δ logloss)"
):
    if df_top is None or df_top.empty or "importance" not in df_top.columns:
        return None
    dfp = df_top.sort_values("importance", ascending=False).reset_index(drop=True)
    values = dfp["importance"].to_numpy()
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
        ax.text(x + offset, ytxt, f"{val:.4f}", va="center", ha="left", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlabel("Δ logloss (permute)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_xlstm_brier_mcc(
    val_md: Dict[str, float],
    test_md: Optional[Dict[str, float]] = None,
    title_suffix: str = "XLSTM (Calibrated)",
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


# -------- permutation importance on validation --------
def _permutation_importance_val(
    model_like,
    iso: Optional[IsotonicRegression],
    X_val_df,
    y_val,
    seed: int = 12345,
) -> pd.DataFrame:
    X_val = _to_float32_df(X_val_df)
    yv = np.asarray(y_val).astype(int)

    # baseline prob (uncal), then calibrate if iso is provided
    p_base = model_like.predict_proba(X_val_df).ravel()
    if iso is not None:
        p_base = np.clip(iso.predict(p_base), 0.0, 1.0)
    base_ll = _safe_log_loss(yv, p_base)

    feats = list(
        getattr(X_val_df, "columns", [f"f{i}" for i in range(X_val.shape[1])])
    )
    imps = []
    rng = np.random.default_rng(seed)

    for j, f in enumerate(feats):
        Xp = X_val.copy()
        rng.shuffle(Xp[:, j])
        p_perm = model_like.predict_proba(Xp).ravel()
        if iso is not None:
            p_perm = np.clip(iso.predict(p_perm), 0.0, 1.0)
        ll_perm = _safe_log_loss(yv, p_perm)
        imps.append(max(0.0, float(ll_perm - base_ll)))

    return (
        pd.DataFrame({"feature": feats, "importance": imps})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# -------- main pipeline --------
def train_validate_test_xlstm(
    X_train_res_scaled,
    y_train_res,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test,
    feature_names: Optional[List[str]] = None,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 120,
    patience: int = 30,
    random_state: int = 42,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,  # 0/None => ALL
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Saves to: CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] / 'xlstm'
    Artifacts:
      - xlstm_metrics_train.csv / xlstm_metrics_val.csv / xlstm_metrics_test.csv
      - xlstm_val_report.txt / xlstm_test_report.txt
      - xlstm_val_summary.csv / xlstm_test_summary.csv (macro_avg_f1 column)
      - xlstm_brier_mcc_summary.csv
      - xlstm_roc.png / xlstm_pr.png / xlstm_calibration.png / xlstm_loss_curve.png
      - xlstm_brier_mcc.png
      - xlstm_features_topK.csv (or _all.csv) + xlstm_features_bar_topK.png (or _all.png)
      - xlstm_summary.txt
    """
    CONFIGS = CONFIGS or {}
    if (
        "XLSTM_TOPN_FEATURES" in CONFIGS
        and CONFIGS["XLSTM_TOPN_FEATURES"] is not None
    ):
        topn_features = CONFIGS["XLSTM_TOPN_FEATURES"]

    outdir = _outdir(CONFIGS) if save_outputs else None
    if outdir is not None:
        _stamp(f"[SAVE] Output directory: {outdir.resolve()}")

    Xtr = _to_float32_df(X_train_res_scaled)
    Xva = _to_float32_df(X_val_scaled)
    Xte = _to_float32_df(X_test_scaled) if X_test_scaled is not None else None

    feature_names = feature_names or list(
        getattr(
            X_train_res_scaled,
            "columns",
            [f"f{i}" for i in range(Xtr.shape[1])],
        )
    )
    pos_w = _pos_weight(np.asarray(y_train_res).astype(int))
    device = _select_device(CONFIGS)

    # fit model
    mdl = XLSTMWrapper(
        input_dim=Xtr.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        random_state=random_state,
        device=device,
        pos_weight=pos_w,
        clip_grad=1.0,
    ).fit(X_train_res_scaled, y_train_res, X_val_scaled, y_val)

    # isotonic calibration on validation (with tiny jitter if needed)
    p_val_uncal = mdl.predict_proba(X_val_scaled)
    p_jitter = p_val_uncal + 1e-8 * np.random.default_rng(42).standard_normal(
        len(p_val_uncal)
    )
    try:
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            p_jitter, np.asarray(y_val).astype(int)
        )
    except Exception:
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            p_val_uncal, np.asarray(y_val).astype(int)
        )
    cal_model = _IsotonicCalibrated(mdl, iso)
    _stamp("[XLSTM] Isotonic calibration done on validation.")

    # threshold tuning on validation
    p_val_cal = cal_model.predict_proba(X_val_scaled)[:, 1]
    best_t = _optimize_threshold(y_val, p_val_cal, metric=threshold_metric)

    # metrics on splits
    out_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # raw
    p_tr_raw = mdl.predict_proba(X_train_res_scaled)
    p_va_raw = p_val_uncal
    p_te_raw = (
        mdl.predict_proba(X_test_scaled) if (X_test_scaled is not None) else None
    )

    out_metrics["train"]["xlstm_raw@0.50"] = _metrics(y_train_res, p_tr_raw, 0.5)
    out_metrics["val"]["xlstm_raw@0.50"] = _metrics(y_val, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["xlstm_raw@0.50"] = _metrics(y_test, p_te_raw, 0.5)

    # calibrated
    p_tr_cal = cal_model.predict_proba(X_train_res_scaled)[:, 1]
    p_te_cal = (
        cal_model.predict_proba(X_test_scaled)[:, 1]
        if (X_test_scaled is not None)
        else None
    )

    out_metrics["train"]["xlstm_cal@0.50"] = _metrics(y_train_res, p_tr_cal, 0.5)
    out_metrics["val"]["xlstm_cal@0.50"] = _metrics(y_val, p_val_cal, 0.5)
    out_metrics["val"][f"xlstm_cal@{best_t:.2f}"] = _metrics(
        y_val, p_val_cal, best_t
    )
    if p_te_cal is not None:
        out_metrics["test"]["xlstm_cal@0.50"] = _metrics(y_test, p_te_cal, 0.5)
        out_metrics["test"][f"xlstm_cal@{best_t:.2f}"] = _metrics(
            y_test, p_te_cal, best_t
        )

    # ---- summaries (for *_summary.csv) ----
    def _summary_df(md: Dict[str, float], threshold: float) -> pd.DataFrame:
        """
        One-row summary; f1 in CSV is macro_avg_f1.
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
            "macro_avg_f1": md.get("macro_f1", md.get("f1", np.nan)),
            "mcc": md.get("mcc", np.nan),
        }
        return pd.DataFrame([row], columns=cols)

    # ---- validation metrics used in report + summary ----
    val_key = f"xlstm_cal@{best_t:.2f}"
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

    if have_test and (p_te_cal is not None):
        test_key = f"xlstm_cal@{best_t:.2f}"
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
        print("\n" + test_report_text)

    # permutation importance on validation (calibrated)
    feat_imp_df = _permutation_importance_val(mdl, iso, X_val_scaled, y_val)
    if (
        topn_features is not None
        and topn_features > 0
        and topn_features < len(feat_imp_df)
    ):
        feat_out = feat_imp_df.head(int(topn_features)).copy()
        used_k: Any = int(topn_features)
        feat_suffix = f"top{used_k}"
    else:
        feat_out = feat_imp_df.copy()
        used_k = "all"
        feat_suffix = "all"
    feat_fig = plot_features_bar(
        feat_out, f"XLSTM Permutation Importance ({feat_suffix})"
    )

    # training / ROC–PR / calibration plots
    loss_fig = plot_xlstm_loss_curves(mdl.history_)
    roc_fig, pr_fig = plot_xlstm_roc_pr(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="XLSTM (Calibrated)",
    )
    calib_fig = plot_xlstm_calibration(
        y_val,
        p_val_cal,
        y_test if have_test else None,
        p_te_cal if have_test else None,
        title_suffix="XLSTM (Calibrated)",
    )

    # summary DataFrames
    val_summary_df = _summary_df(val_md, best_t)

    test_summary_df = pd.DataFrame()
    if have_test and (p_te_cal is not None):
        if test_md is None:
            test_md = _metrics(y_test, p_te_cal, best_t)
        test_summary_df = _summary_df(test_md, best_t)

    # Brier/MCC summary
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
    brier_mcc_fig = plot_xlstm_brier_mcc(
        val_md,
        test_md if (have_test and test_md is not None) else None,
        title_suffix="XLSTM (Calibrated)",
    )

    # save everything
    if outdir is not None:
        if out_metrics["train"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["train"]),
                outdir / "xlstm_metrics_train",
            )
        if out_metrics["val"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["val"]),
                outdir / "xlstm_metrics_val",
            )
        if out_metrics["test"]:
            _save_csv(
                _metrics_split_to_df(out_metrics["test"]),
                outdir / "xlstm_metrics_test",
            )

        _save_csv(feat_out, outdir / f"xlstm_features_{feat_suffix}")
        if feat_fig is not None:
            _save_fig(feat_fig, outdir / f"xlstm_features_bar_{feat_suffix}")

        _save_text(val_report_text, outdir / "xlstm_val_report")
        if test_report_text:
            _save_text(test_report_text, outdir / "xlstm_test_report")

        _save_csv(val_summary_df, outdir / "xlstm_val_summary")
        if not test_summary_df.empty:
            _save_csv(test_summary_df, outdir / "xlstm_test_summary")

        _save_csv(brier_mcc_df, outdir / "xlstm_brier_mcc_summary")

        _save_text(
            "XLSTM hyperparams & run summary:\n"
            f"hidden_dim={hidden_dim}\n"
            f"num_layers={num_layers}\n"
            f"dropout={dropout}\n"
            f"lr={lr}\n"
            f"batch_size(final)={mdl.batch_size}\n"
            f"max_epochs={max_epochs}\n"
            f"patience={patience}\n"
            f"device_final={mdl.device}\n"
            f"best_threshold={best_t:.3f}\n",
            outdir / "xlstm_summary",
        )

        _save_fig(roc_fig, outdir / "xlstm_roc")
        _save_fig(pr_fig, outdir / "xlstm_pr")
        _save_fig(calib_fig, outdir / "xlstm_calibration")
        if loss_fig is not None:
            _save_fig(loss_fig, outdir / "xlstm_loss_curve")
        if brier_mcc_fig is not None:
            _save_fig(brier_mcc_fig, outdir / "xlstm_brier_mcc")

    return {
        "xlstm_model": mdl,
        "xlstm_cal": _IsotonicCalibrated(mdl, iso),
        "best_threshold": best_t,
        "metrics": out_metrics,
        "history": mdl.history_,
        "features": feat_out,
        "val_report_text": val_report_text,
        "test_report_text": test_report_text,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
        "brier_mcc_df": brier_mcc_df,
        "used_topn_features": used_k,
        "outdir": str(outdir) if outdir is not None else None,
    }
