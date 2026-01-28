# utils/pipeline_node.py
# ============================
# NODE Pipeline (train/validate/test + save)
# Pure PyTorch implementation of Neural Oblivious Decision Ensembles.
# ============================
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as Fnn  # <- DO NOT SHADOW
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    log_loss, accuracy_score, f1_score, brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve

# ---------- small IO helpers ----------
def _stamp(msg: str): print(msg)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _outdir(CONFIGS: Dict[str, Any]) -> Path:
    p = Path(CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR']) / "node"
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

# ---------- generic helpers ----------
def _predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 1: p = np.column_stack([1 - p, p])
        elif p.shape[1] == 1: p = np.column_stack([1 - p[:, 0], p[:, 0]])
        return p
    raise AttributeError("predict_proba required for this call")

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

def _metrics(y_true, p, t: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p).reshape(-1)
    y_pred = (p1 >= t).astype(int)
    out = {
        "auc": roc_auc_score(y_true, p1) if len(np.unique(y_true)) == 2 else float('nan'),
        "ap":  average_precision_score(y_true, p1),
        "logloss": log_loss(y_true, p1, labels=[0,1]),
        "brier": brier_score_loss(y_true, p1),
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred),
    }
    return {k: float(v) for k, v in out.items()}

def _metrics_split_to_df(split_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, md in split_dict.items():
        row = {"model": name}
        row.update(md)
        rows.append(row)
    return pd.DataFrame(rows)

def _optimize_threshold(y_true, p, *, metric: str = "f1", print_fn=None) -> float:
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
    if print_fn:
        print_fn(f"[NODE][THRESH] best_t={best_t:.3f} ({metric}={best_s:.4f})")
    return float(best_t)

# ---------- isotonic calibrator wrapper ----------
class _IsotonicCalibrated:
    def __init__(self, base_estimator, iso: IsotonicRegression):
        self.base_estimator = base_estimator
        self.iso = iso
        self.classes_ = np.array([0, 1], dtype=int)
    def predict_proba(self, X):
        p = self.base_estimator.predict_proba(X)
        pc = np.clip(self.iso.predict(p[:, 1]), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ---------- NODE core (PyTorch) ----------
def _device_from_configs(CONFIGS):
    if CONFIGS and CONFIGS.get("USE_GPU", True) and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _leaf_codes(depth: int, device):
    L = 1 << depth
    codes = []
    for i in range(L):
        bits = [(i >> k) & 1 for k in range(depth)]
        codes.append(bits)
    return torch.tensor(codes, dtype=torch.float32, device=device)  # (L, D)

class _NodeTree(nn.Module):
    def __init__(self, n_features: int, depth: int, device):
        super().__init__()
        self.n_features = int(n_features)
        self.depth = int(depth)
        self.device = device

        # all learnables are explicitly float32
        self.alpha = nn.Parameter(torch.empty(self.depth, self.n_features, dtype=torch.float32))
        nn.init.xavier_uniform_(self.alpha)

        self.theta = nn.Parameter(torch.zeros(self.depth, dtype=torch.float32))
        self.tau_raw = nn.Parameter(torch.full((self.depth,), -1.0, dtype=torch.float32))  # softplus -> ~0.3

        L = 1 << self.depth
        self.leaf_logits = nn.Parameter(torch.zeros(L, dtype=torch.float32))

        self.register_buffer("leaf_codes", _leaf_codes(self.depth, device))  # (L, D), no grad

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, F) float32
        B = X.shape[0]
        D = self.depth

        W = Fnn.softmax(self.alpha, dim=1)                  # (D, F)
        S = X @ W.T                                         # (B, D)
        tau = Fnn.softplus(self.tau_raw) + 1e-3             # (D,)
        g = torch.sigmoid((S - self.theta) / tau)           # (B, D)

        g_exp = g.unsqueeze(1)                              # (B, 1, D)
        code = self.leaf_codes.unsqueeze(0)                 # (1, L, D)
        p = torch.where(code > 0.5, g_exp, 1.0 - g_exp)     # (B, L, D)
        p_leaf = p.prod(dim=2)                              # (B, L)

        out = p_leaf @ self.leaf_logits                     # (B,)
        return out

    def feature_importance(self) -> torch.Tensor:
        with torch.no_grad():
            W = Fnn.softmax(self.alpha, dim=1)  # (D, F)
            return W.sum(dim=0)                 # (F,)

class _NodeEnsemble(nn.Module):
    def __init__(self, n_features: int, num_trees: int, depth: int, device):
        super().__init__()
        self.trees = nn.ModuleList([_NodeTree(n_features, depth, device) for _ in range(int(num_trees))])
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    def forward(self, X):
        s = self.bias.expand(X.size(0))
        for t in self.trees:
            s = s + t(X)
        return s  # logits
    def feature_importance(self) -> torch.Tensor:
        with torch.no_grad():
            imps = [t.feature_importance() for t in self.trees]
            return torch.stack(imps, dim=0).sum(dim=0)

class _NodeClassifier:
    """Scikit-like wrapper around NODE ensemble."""
    def __init__(
        self, n_features: int, num_trees: int = 64, depth: int = 3,
        lr: float = 1e-3, batch_size: int = 256, max_epochs: int = 120,
        patience: int = 40, random_state: int = 42, device: str = "cpu",
        pos_weight: Optional[float] = None, logger=None
    ):
        self.n_features = int(n_features)
        self.num_trees = int(num_trees)
        self.depth = int(depth)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device = device
        self.pos_weight = pos_weight
        self.logger = logger or (lambda _msg: None)  # quiet by default

        torch.manual_seed(self.random_state)
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model = _NodeEnsemble(self.n_features, self.num_trees, self.depth, self.device).to(self.device)
        self.history_: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_logloss": [], "val_auc": []}
        self._best_state = None

    def _to_tensor(self, X) -> torch.Tensor:
        Xn = X.values if hasattr(X, "values") else np.asarray(X)
        return torch.tensor(Xn, dtype=torch.float32, device=self.device)

    def _predict_logits(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xb = self._to_tensor(X)
            out = self.model(Xb)
            return out.detach().cpu().numpy()

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.model.train()
        Xtr = self._to_tensor(X_train)
        ytr = torch.tensor(np.asarray(y_train).astype(np.float32), device=self.device)
        Xva = self._to_tensor(X_val)
        yva_np = np.asarray(y_val).astype(int)

        ds = torch.utils.data.TensorDataset(Xtr, ytr)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos_w = None
        if self.pos_weight is not None and self.pos_weight > 0:
            pos_w = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        best_ll = float("inf")
        best_auc = -1.0
        no_improve = 0

        for ep in range(1, self.max_epochs + 1):
            self.model.train()
            ep_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * xb.size(0)
            ep_loss /= len(ds)

            # val
            self.model.eval()
            with torch.no_grad():
                logits_val = self.model(Xva).detach().cpu().numpy().reshape(-1)
                p_val = 1 / (1 + np.exp(-logits_val))
                v_ll = log_loss(yva_np, p_val, labels=[0,1])
                v_auc = roc_auc_score(yva_np, p_val) if len(np.unique(yva_np)) == 2 else float("nan")

            self.history_["epoch"].append(ep)
            self.history_["train_loss"].append(ep_loss)
            self.history_["val_logloss"].append(v_ll)
            self.history_["val_auc"].append(v_auc)
            self.logger(f"[NODE] epoch={ep:03d} train_loss={ep_loss:.4f} val_logloss={v_ll:.4f} val_auc={v_auc:.4f}")

            improved = (v_ll < best_ll - 1e-9) or (abs(v_ll - best_ll) <= 1e-9 and v_auc > best_auc)
            if improved:
                best_ll = v_ll
                best_auc = v_auc
                no_improve = 0
                self._best_state = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= self.patience:
                self.logger(f"[NODE] Early stopping at epoch {ep} (best_ll={best_ll:.5f})")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        logits = self._predict_logits(X).reshape(-1)
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1.0 - p, p])

    def feature_importances_(self, feature_names: List[str], topn: Optional[int] = 30) -> pd.DataFrame:
        with torch.no_grad():
            imp = self.model.feature_importance().detach().cpu().numpy().reshape(-1)
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": imp,
            "abs_importance": np.abs(imp),
        }).sort_values("abs_importance", ascending=False).reset_index(drop=True)
        if topn and topn > 0:
            df = df.head(int(topn)).copy()
        return df

# ---------- plots ----------
def plot_node_roc_pr(y_val, p_val, y_test=None, p_test=None, title_suffix="NODE (calibrated)"):
    figs = []
    fig = plt.figure()
    fpr_v, tpr_v, _ = roc_curve(y_val, p_val); auc_v = roc_auc_score(y_val, p_val)
    plt.plot(fpr_v, tpr_v, label=f"Val (AUC={auc_v:.3f})")
    if y_test is not None and p_test is not None:
        fpr_t, tpr_t, _ = roc_curve(y_test, p_test); auc_t = roc_auc_score(y_test, p_test)
        plt.plot(fpr_t, tpr_t, label=f"Test (AUC={auc_t:.3f})")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.title(f"ROC — {title_suffix}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(True); plt.legend(); plt.tight_layout()
    figs.append(fig)
    fig = plt.figure()
    prec_v, rec_v, _ = precision_recall_curve(y_val, p_val); ap_v = average_precision_score(y_val, p_val)
    plt.plot(rec_v, prec_v, label=f"Val (AP={ap_v:.3f})")
    if y_test is not None and p_test is not None:
        prec_t, rec_t, _ = precision_recall_curve(y_test, p_test); ap_t = average_precision_score(y_test, p_test)
        plt.plot(rec_t, prec_t, label=f"Test (AP={ap_t:.3f})")
    plt.title(f"Precision–Recall — {title_suffix}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True); plt.legend(); plt.tight_layout()
    figs.append(fig)
    return figs

def plot_node_calibration(y_val, p_val, y_test=None, p_test=None, n_bins=15, title_suffix="NODE (calibrated)"):
    fig = plt.figure()
    frac_v, mean_v = calibration_curve(y_val, p_val, n_bins=n_bins, strategy="quantile")
    plt.plot(mean_v, frac_v, marker="o", label="Validation")
    if y_test is not None and p_test is not None:
        frac_t, mean_t = calibration_curve(y_test, p_test, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_t, frac_t, marker="s", label="Test")
    line = np.linspace(0,1,100); plt.plot(line, line, "--", color="gray")
    plt.title(f"Reliability Diagram — {title_suffix}")
    plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction of positives")
    plt.grid(True); plt.legend(); plt.tight_layout()
    return fig

def plot_node_loss(history: Dict[str, List[float]]):
    if not history or not history.get("epoch"):
        return None
    fig = plt.figure()
    if history.get("train_loss"):
        plt.plot(history["epoch"], history["train_loss"], label="train loss")
    if history.get("val_logloss"):
        plt.plot(history["epoch"], history["val_logloss"], label="val logloss")
    plt.title("NODE: training vs validation")
    plt.xlabel("epoch"); plt.grid(True); plt.legend(); plt.tight_layout()
    return fig

def plot_top_features_bar(df_top: pd.DataFrame, title="NODE Top Features"):
    if df_top is None or df_top.empty: return None
    fig = plt.figure(figsize=(8, max(5, 0.32 * len(df_top))))
    y = np.arange(len(df_top))
    vals = df_top["abs_importance"].values
    plt.barh(y, vals)
    plt.yticks(y, df_top["feature"].values)
    plt.gca().invert_yaxis()  # largest on top
    for i, v in enumerate(vals):
        plt.text(v, i, f" {v:.3g}", va="center")
    plt.xlabel("|importance|"); plt.title(title); plt.tight_layout()
    return fig

# ---------- main pipeline ----------
def train_validate_test_node(
    X_train_res_scaled, y_train_res,
    X_val_scaled, y_val,
    X_test_scaled, y_test,
    feature_names: Optional[List[str]] = None,
    *,
    num_trees_grid: Optional[List[int]] = None,
    depth_grid: Optional[List[int]] = None,
    lr_grid: Optional[List[float]] = None,
    batch_size_grid: Optional[List[int]] = None,
    max_epochs: int = 120,
    patience: int = 40,
    random_state: int = 42,
    threshold_metric: str = "f1",
    topn_features: Optional[int] = 30,   # 0/None => ALL
    CONFIGS: Optional[Dict[str, Any]] = None,
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Full NODE pipeline on *scaled* splits with isotonic calibration & thresholding.

    Saves under: CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] / 'node'
      - node_metrics_train.csv, node_metrics_val.csv, node_metrics_test.csv
      - node_val_report.txt, node_test_report.txt
      - node_val_summary.csv, node_test_summary.csv
      - node_roc.png, node_pr.png, node_calibration.png, node_loss_curve.png
      - node_top{N}.csv (or node_topALL.csv), node_top_features_bar.png
    """
    assert CONFIGS is not None and "DIR_tr_va_te_metric_shap_SAVE_DIR" in CONFIGS, "CONFIGS['DIR_tr_va_te_metric_shap_SAVE_DIR'] required"
    outdir = _outdir(CONFIGS) if save_outputs else None
    if outdir is not None:
        _stamp(f"[SAVE] Output directory: {outdir.resolve()}")

    # ---- console control ----
    console_mode = (CONFIGS or {}).get("NODE_CONSOLE_MODE", "oom_only")  # "oom_only" | "all" | "none"
    def _log(msg: str):
        if console_mode == "all":
            print(msg)
    def _log_oom(msg: str):
        if console_mode in ("all", "oom_only"):
            print(msg)
    def _noop(_msg: str):  # for threshold print suppression
        pass

    # ---- prep ----
    feature_names = feature_names or list(getattr(X_train_res_scaled, "columns",
                                                  [f"f{i}" for i in range(X_train_res_scaled.shape[1])]))
    device_pref = "cuda" if (CONFIGS.get("USE_GPU", True) and torch.cuda.is_available()) else "cpu"
    _log(f"[NODE] Device preference: {device_pref} (cuda_available={torch.cuda.is_available()})")

    num_trees_grid = num_trees_grid or [64, 128]
    depth_grid     = depth_grid     or [3, 4]
    lr_grid        = lr_grid        or [1e-3, 3e-4]
    batch_size_grid= batch_size_grid or [256, 512]

    ytr = np.asarray(y_train_res).astype(int)
    pos = (ytr == 1).sum()
    neg = (ytr == 0).sum()
    pos_w = (neg / (pos + 1e-9)) if (pos > 0 and neg > 0) else None

    best = None
    best_hist = {}
    _log("[NODE] Hyperparam scan (num_trees / depth / lr / batch_size):")
    for nt in num_trees_grid:
        for dp in depth_grid:
            for lr in lr_grid:
                for bs in batch_size_grid:
                    for dev in ([device_pref, "cpu"] if device_pref == "cuda" else ["cpu"]):
                        try:
                            mdl = _NodeClassifier(
                                n_features=len(feature_names),
                                num_trees=int(nt), depth=int(dp),
                                lr=float(lr), batch_size=int(bs),
                                max_epochs=int(max_epochs), patience=int(patience),
                                random_state=int(random_state),
                                device=("cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu"),
                                pos_weight=pos_w,
                                logger=(_log if console_mode == "all" else (lambda _m: None))
                            ).fit(X_train_res_scaled, y_train_res, X_val_scaled, y_val)
                            p_val = mdl.predict_proba(X_val_scaled)[:, 1]
                            v_ll  = log_loss(y_val, p_val, labels=[0,1])
                            v_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) == 2 else float("nan")
                            _log(f"  nt={nt:<3}  depth={dp:<1}  lr={lr:<.0e}  bs={bs:<3}  dev={dev:<4}  val_logloss={v_ll:.4f}  val_auc={v_auc:.4f}")
                            if (best is None) or (v_ll < best["val_logloss"] - 1e-9) or (abs(v_ll - best["val_logloss"]) <= 1e-9 and v_auc > best["val_auc"]):
                                best = {"nt": nt, "depth": dp, "lr": lr, "bs": bs, "device": dev, "model": mdl, "val_logloss": v_ll, "val_auc": v_auc}
                                best_hist = mdl.history_
                            break
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                                _log_oom(f"[NODE] CUDA OOM for nt={nt}, depth={dp}, bs={bs}. Falling back to CPU for this combo.")
                                continue
                            else:
                                # fully suppress other combo errors unless console_mode="all"
                                _log(f"[NODE] Skipping combo nt={nt}, depth={dp}, lr={lr}, bs={bs} due to error: {e}")
                                continue

    if best is None:
        raise RuntimeError("NODE fit failed for all scanned configurations.")

    node_raw = best["model"]
    _log(f"[NODE] Best: num_trees={best['nt']}, depth={best['depth']}, lr={best['lr']}, batch_size={best['bs']} (device={best['device']}).")

    # ---- isotonic calibration on validation ----
    p_val_uncal = node_raw.predict_proba(X_val_scaled)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_uncal, np.asarray(y_val).astype(int))
    node_cal = _IsotonicCalibrated(node_raw, iso)
    _log("[NODE] Isotonic calibration done on validation.")

    # ---- threshold selection ----
    p_val_cal = node_cal.predict_proba(X_val_scaled)[:, 1]
    best_t = _optimize_threshold(y_val, p_val_cal, metric=threshold_metric,
                                 print_fn=(print if console_mode == "all" else _noop))

    # ---- metrics ----
    out_metrics = {"train": {}, "val": {}, "test": {}}
    p_tr_raw = node_raw.predict_proba(X_train_res_scaled)[:, 1]
    p_va_raw = p_val_uncal
    p_te_raw = node_raw.predict_proba(X_test_scaled)[:, 1] if (X_test_scaled is not None) else None

    out_metrics["train"]["node_raw@0.50"] = _metrics(y_train_res, p_tr_raw, 0.5)
    out_metrics["val"]["node_raw@0.50"]   = _metrics(y_val, p_va_raw, 0.5)
    if p_te_raw is not None:
        out_metrics["test"]["node_raw@0.50"]  = _metrics(y_test, p_te_raw, 0.5)

    p_tr_cal = node_cal.predict_proba(X_train_res_scaled)[:, 1]
    p_te_cal = node_cal.predict_proba(X_test_scaled)[:, 1] if (X_test_scaled is not None) else None

    out_metrics["train"]["node_cal@0.50"] = _metrics(y_train_res, p_tr_cal, 0.5)
    out_metrics["val"]["node_cal@0.50"]   = _metrics(y_val, p_val_cal, 0.5)
    out_metrics["val"][f"node_cal@{best_t:.2f}"] = _metrics(y_val, p_val_cal, best_t)
    if p_te_cal is not None:
        out_metrics["test"]["node_cal@0.50"]      = _metrics(y_test, p_te_cal, 0.5)
        out_metrics["test"][f"node_cal@{best_t:.2f}"] = _metrics(y_test, p_te_cal, best_t)

    # ---- reports (PRINT + SAVE) ----
    val_report_text = (
        f"Best threshold (F1 on VAL): {best_t:.2f}\n\n"
        f"🔎 Validation Accuracy (threshold={best_t:.2f}): "
        f"{accuracy_score(y_val, (p_val_cal>=best_t).astype(int)):.2f}\n"
        f"Classification Report (Validation):\n"
        f"{classification_report(y_val, (p_val_cal>=best_t).astype(int), target_names=['Class 0','Class 1'], digits=2)}\n"
        f"AUC-ROC (Validation): {roc_auc_score(y_val, p_val_cal):.2f}"
    )
    print("\n" + val_report_text)

    test_report_text = ""
    if X_test_scaled is not None and p_te_cal is not None:
        y_pred_test = (p_te_cal >= best_t).astype(int)
        test_report_text = (
            f"Best threshold (F1): {best_t:.2f}\n\n"
            f"🔎 Test Accuracy (threshold={best_t:.2f}): {accuracy_score(y_test, y_pred_test):.2f}\n"
            f"Classification Report:\n"
            f"{classification_report(y_test, y_pred_test, target_names=['Class 0','Class 1'], digits=2)}\n"
            f"AUC-ROC (Test): {roc_auc_score(y_test, p_te_cal):.2f}"
        )
        print("\n" + test_report_text)

    # ---- features ----
    topn = None if (topn_features is None or topn_features == 0) else int(topn_features)
    feat_df = node_raw.feature_importances_(feature_names, topn=topn)

    # ---- plots ----
    loss_fig  = plot_node_loss(best_hist)
    roc_fig, pr_fig = plot_node_roc_pr(y_val, p_val_cal,
                                       y_test if X_test_scaled is not None else None,
                                       p_te_cal if X_test_scaled is not None else None,
                                       title_suffix="NODE (Calibrated)")
    calib_fig = plot_node_calibration(y_val, p_val_cal,
                                      y_test if X_test_scaled is not None else None,
                                      p_te_cal if X_test_scaled is not None else None,
                                      title_suffix="NODE (Calibrated)")
    top_bar_fig = plot_top_features_bar(feat_df)

    # ---- summaries (CSV rows @ best_t) ----
    def _row(md: Dict[str, float], split: str) -> pd.DataFrame:
        r = dict(split=split, threshold=best_t, **md)
        return pd.DataFrame([r])
    val_md = out_metrics["val"].get(f"node_cal@{best_t:.2f}") or _metrics(y_val, p_val_cal, best_t)
    val_summary_df = _row(val_md, "val")
    test_summary_df = pd.DataFrame()
    if X_test_scaled is not None and p_te_cal is not None:
        test_md = out_metrics["test"].get(f"node_cal@{best_t:.2f}") or _metrics(y_test, p_te_cal, best_t)
        test_summary_df = _row(test_md, "test")

    # ---- SAVE ----
    if outdir is not None:
        if out_metrics["train"]:
            _save_csv(_metrics_split_to_df(out_metrics["train"]), outdir / "node_metrics_train")
        if out_metrics["val"]:
            _save_csv(_metrics_split_to_df(out_metrics["val"]),   outdir / "node_metrics_val")
        if out_metrics["test"]:
            _save_csv(_metrics_split_to_df(out_metrics["test"]),  outdir / "node_metrics_test")

        _save_text(val_report_text, outdir / "node_val_report")
        if test_report_text:
            _save_text(test_report_text, outdir / "node_test_report")

        _save_csv(val_summary_df, outdir / "node_val_summary")
        if not test_summary_df.empty:
            _save_csv(test_summary_df, outdir / "node_test_summary")

        _save_csv(feat_df, outdir / (f"node_top{len(feat_df)}" if topn else "node_topALL"))

        _save_fig(roc_fig,   outdir / "node_roc")
        _save_fig(pr_fig,    outdir / "node_pr")
        _save_fig(calib_fig, outdir / "node_calibration")
        if loss_fig is not None:
            _save_fig(loss_fig, outdir / "node_loss_curve")
        if top_bar_fig is not None:
            _save_fig(top_bar_fig, outdir / "node_top_features_bar")

    return {
        "node_raw": node_raw,
        "node_cal": node_cal,
        "best_params": {"num_trees": best["nt"], "depth": best["depth"], "lr": best["lr"], "batch_size": best["bs"]},
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
