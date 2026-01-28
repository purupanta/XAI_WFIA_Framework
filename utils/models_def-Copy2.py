from __future__ import annotations

from utils.helpers import logging_helpers

__all__ = [
    "train_models",
    "fit_lr_safe",
    "fit_rf_safe",
    "fit_xgb_safe",
    "fit_mlp_safe",
    "fit_tabnet_safe",
    "calibrate_model_isotonic",
    "AveragingEnsemble",
    "plot_metric_curves",
    "plot_train_val_loss",
    "plot_roc_all",
    "plot_pr_all",
    "optimize_threshold",
]

# ========= Safe fit helpers & plotting =========
import warnings
from typing import Dict, Tuple, Optional, Any, Iterable, Callable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, log_loss
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except Exception:  # if xgboost optional/missing
    XGBClassifier = None

def stamp(msg: str):
    print(msg)

# ----- Shared utilities -----
def _balanced_sample_weight(y):
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)
    w_pos = 0.5 / pos
    w_neg = 0.5 / neg
    sw = np.where(y == 1, w_pos, w_neg)
    return (sw * y.size).astype(np.float32)  # normalize ~N

def _pos_weight(y):
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return (neg / max(pos, 1)) if pos > 0 else 1.0

def _to_numpy(X):
    return X.values if hasattr(X, "values") else np.asarray(X)

# ========= MAIN CALLING FUNCTION =========
def train_models(
    CONFIGS: Dict[str, Any],
    X_train_res_scaled, y_train_res,
    X_train_res, X_val, y_val,
    X_test=None, y_test=None,
    models: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains LR, RF, XGB, MLP, TabNet.
    Returns (models, histories). histories contains per-iteration metrics where available.
    Adds 'xgb_cal' (isotonic) and 'blend' ensemble if possible.
    """
    if models is None:
        models = {}

    histories: Dict[str, Any] = {}

    # ===== Logistic Regression =====
    with logging_helpers.Timer("Train Logistic Regression"):
        models["lr"] = fit_lr_safe(
            X_train_res_scaled, y_train_res,
            max_iter=600, class_weight="balanced",
            random_state=CONFIGS['RANDOM_STATE'], prefer_saga=True
        )

    # ===== Random Forest =====
    with logging_helpers.Timer("Train Random Forest"):
        models["rf"] = fit_rf_safe(
            X_train_res_scaled, y_train_res,
            n_estimators=int(CONFIGS.get('RF_TREES', 700)),
            max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=CONFIGS['RANDOM_STATE']
        )

    # ===== XGBoost =====
    with logging_helpers.Timer("Train XGBoost"):
        models["xgb"], histories["xgb"] = fit_xgb_safe(
            X_train_res_scaled, y_train_res,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            max_depth=int(CONFIGS.get("XGB_MAX_DEPTH", 6)),
            min_child_weight=float(CONFIGS.get("XGB_MIN_CHILD_WEIGHT", 3.0)),
            subsample=float(CONFIGS.get("XGB_SUBSAMPLE", 0.8)),
            colsample_bytree=float(CONFIGS.get("XGB_COLSAMPLE", 0.7)),
            n_estimators=int(CONFIGS.get("XGB_TREES", 2500)),
            learning_rate=float(CONFIGS.get("XGB_LR", 0.03)),
            random_state=CONFIGS['RANDOM_STATE']
        )

    # ===== MLP =====
    with logging_helpers.Timer("Train MLP"):
        models["mlp"], histories["mlp"] = fit_mlp_safe(
            X_train_res_scaled, y_train_res,
            X_val=X_val if X_val is not None else None,
            y_val=y_val if y_val is not None else None,
            hidden_layer_sizes=tuple(CONFIGS.get('MLP_HIDDEN', (128, 64))),
            random_state=CONFIGS['RANDOM_STATE']
        )

    # ===== TabNet =====
    with logging_helpers.Timer("Train TabNet"):
        models["tabnet"], histories["tabnet"] = fit_tabnet_safe(
            X_train_res, y_train_res,
            X_val, y_val,
            X_test=X_test, y_test=y_test,
            width=int(CONFIGS.get("TABNET_WIDTH", 24)),
            n_steps=int(CONFIGS.get("TABNET_STEPS", 4)),
            max_epochs=int(CONFIGS.get("TABNET_EPOCHS", 120)),
            patience=int(CONFIGS.get("TABNET_PATIENCE", 40)),
            batch_size=int(CONFIGS.get("TABNET_BS", 256)),
            virtual_batch_size=int(CONFIGS.get("TABNET_VBS", 32)),
            lr=float(CONFIGS.get("TABNET_LR", 6e-4)),
            gamma=float(CONFIGS.get("TABNET_GAMMA", 1.5)),
            lambda_sparse=float(CONFIGS.get("TABNET_LAMBDA_SPARSE", 1e-5)),
            mask_type=str(CONFIGS.get("TABNET_MASK", "sparsemax")),
            random_state=CONFIGS['RANDOM_STATE'],
            use_gpu=bool(CONFIGS.get("TABNET_USE_GPU", True)),
            cat_idxs=CONFIGS.get("CAT_IDXS", []),
            cat_dims=CONFIGS.get("CAT_DIMS", []),
            cat_emb_dim=CONFIGS.get("CAT_EMB_DIM", 8),
            es_min_delta=float(CONFIGS.get("TABNET_ES_MIN_DELTA", 5e-4)),
            es_monitor=str(CONFIGS.get("TABNET_ES_MONITOR", "val_0_auc")),
            es_mode=str(CONFIGS.get("TABNET_ES_MODE", "max")),
            use_pretrain=bool(CONFIGS.get("TABNET_USE_PRETRAIN", False)),
            pretrain_epochs=int(CONFIGS.get("TABNET_PRE_EPOCHS", 100)),
            pretrain_patience=int(CONFIGS.get("TABNET_PRE_PATIENCE", 20)),
            pretrain_bs=int(CONFIGS.get("TABNET_PRE_BS", 512)),
            pretraining_ratio=float(CONFIGS.get("TABNET_PRE_RATIO", 0.8)),
        )

    # ===== Calibrate XGB (isotonic) =====
    with logging_helpers.Timer("Calibrate XGBoost (isotonic)"):
        try:
            models["xgb_cal"] = calibrate_model_isotonic(models["xgb"], X_val, y_val)
            stamp("[CAL] Calibrated XGB with isotonic regression.")
        except Exception as e:
            stamp(f"[CAL] Calibration skipped: {e}")

    # ===== Weighted ensemble =====
    with logging_helpers.Timer("Build weighted ensemble"):
        try:
            weight_sources = ["xgb_cal" if "xgb_cal" in models else "xgb", "rf", "lr"]
            w = _weights_from_val_auc(models, weight_sources, X_val, y_val)
            models["blend"] = AveragingEnsemble({k: models[k] for k in weight_sources}, weights=w)
            stamp(f"[ENSEMBLE] Using weights {w}")
        except Exception as e:
            stamp(f"[ENSEMBLE] Skipped: {e}")

    return models, histories


# ========= LR =========
def fit_lr_safe(
    X, y, *,
    C=1.0, penalty="l2", l1_ratio=None,
    max_iter=400, class_weight="balanced",
    random_state=0, prefer_saga=True
):
    solvers = (["saga","lbfgs","liblinear"] if prefer_saga else ["lbfgs","saga","liblinear"])
    last_err = None
    for solver in solvers:
        if penalty == "elasticnet" and solver != "saga":
            continue
        if penalty in ("l1","elasticnet") and solver == "lbfgs":
            continue

        params = dict(
            C=float(C), penalty=penalty, l1_ratio=l1_ratio,
            max_iter=max_iter, class_weight=class_weight,
            random_state=random_state, solver=solver,
            n_jobs=-1 if solver=="saga" else None,
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                lr = LogisticRegression(**{k:v for k,v in params.items() if v is not None})
                lr.fit(X, y)
                if any(isinstance(wi.message, ConvergenceWarning) for wi in w):
                    stamp(f"[LR/{solver}] ConvergenceWarning; increasing max_iter to {max_iter*2}.")
                    lr.set_params(max_iter=max_iter*2, warm_start=True).fit(X, y)
            stamp(f"[LR] Using solver='{solver}', penalty='{penalty}', C={C}, l1_ratio={l1_ratio}.")
            return lr
        except Exception as e:
            last_err = e
            stamp(f"[LR/{solver}] Failed; trying next solver. Reason: {e}")
    raise last_err

# ========= RF =========
def fit_rf_safe(
    X, y, *,
    n_estimators=700, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced_subsample", max_samples=None,
    random_state=0
):
    def _try(n_trees, depth):
        rf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, class_weight=class_weight,
            n_jobs=-1, random_state=random_state, bootstrap=True, oob_score=False,
            max_samples=max_samples
        )
        rf.fit(X, y)
        return rf
    try:
        rf = _try(n_estimators, max_depth)
        stamp(f"[RF] Trained with {n_estimators} trees, depth={max_depth}, leaf={min_samples_leaf}.")
        return rf
    except Exception as e:
        stamp(f"[RF] Error; retrying smaller. Reason: {e}")
        n_small = max(300, n_estimators // 2)
        d_small = 20 if (max_depth is None or max_depth > 20) else max_depth
        rf = _try(n_small, d_small)
        stamp(f"[RF] Using fallback {n_small} trees, depth={d_small}.")
        return rf

# ========= XGB =========
def fit_xgb_safe(
    X, y, *,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    n_estimators=2500, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
    min_child_weight=3.0,
    random_state=0
):
    """
    Returns (model, history). History contains train/val(/test) logloss & AUC (if val/test provided).
    """
    history = {"x": []}
    X = _to_numpy(X); y = np.asarray(y)

    if XGBClassifier is None:
        stamp("[XGB] xgboost not installed — skipping.")
        return None, history

    spw = _pos_weight(y)

    eval_set = [(X, y)]
    if X_val is not None and y_val is not None:
        eval_set.append((_to_numpy(X_val), np.asarray(y_val)))
    if X_test is not None and y_test is not None:
        eval_set.append((_to_numpy(X_test), np.asarray(y_test)))

    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        eval_metric=["auc", "logloss"],   # only in ctor
        tree_method="hist",
        predictor="cpu_predictor",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    try:
        xgb.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=200 if (X_val is not None and y_val is not None) else None
        )
    except ValueError as e:
        if "eval_metric" in str(e):
            stamp("[XGB] Retrying with single eval_metric='auc' for compatibility.")
            xgb.set_params(eval_metric="auc")
            xgb.fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=200 if (X_val is not None and y_val is not None) else None
            )
        else:
            raise

    stamp("[XGB] Trained (CPU, hist) with early stopping." if len(eval_set) > 1 else "[XGB] Trained (CPU, hist).")

    er = xgb.evals_result()
    max_len = 0
    for metrics in er.values():
        for series in metrics.values():
            max_len = max(max_len, len(series))
    history["x"] = list(range(1, max_len + 1))
    for split_name, metrics in er.items():
        for metric_name, series in metrics.items():
            history[f"{split_name}_{metric_name}"] = series

    return xgb, history

# ========= MLP =========
def fit_mlp_safe(
    X, y, *,
    X_val=None, y_val=None,  # only used for reporting (MLP uses internal val split)
    hidden_layer_sizes=(128,64), activation="relu", solver="adam",
    alpha=2e-4, learning_rate_init=8e-4, max_iter=400,
    early_stopping=True, n_iter_no_change=20, class_weight="balanced",
    validation_fraction=0.15,
    random_state=0
):
    """
    Returns (model, history) with training loss, internal validation scores (if early_stopping).
    """
    # sample weights if supported
    sample_weight = None
    if class_weight == "balanced":
        sample_weight = _balanced_sample_weight(y)

    # check whether this sklearn supports sample_weight
    import inspect
    def _supports_sample_weight():
        try:
            sig = inspect.signature(MLPClassifier.fit)
            return "sample_weight" in sig.parameters
        except Exception:
            return False
    sw_ok = _supports_sample_weight()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
            alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter,
            early_stopping=early_stopping, n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction, random_state=random_state
        )
        if sw_ok and sample_weight is not None:
            mlp.fit(X, y, sample_weight=sample_weight)
        else:
            if (sample_weight is not None) and (not sw_ok):
                stamp("[MLP] sample_weight not supported by this sklearn; training without it.")
            mlp.fit(X, y)
        if any(isinstance(wi.message, ConvergenceWarning) for wi in w):
            stamp(f"[MLP] ConvergenceWarning; increasing max_iter to {max_iter*2}.")
            if sw_ok and sample_weight is not None:
                mlp.set_params(max_iter=max_iter*2, warm_start=True).fit(X, y, sample_weight=sample_weight)
            else:
                mlp.set_params(max_iter=max_iter*2, warm_start=True).fit(X, y)

    # Build history
    hist = {"x": list(range(1, len(getattr(mlp, "loss_curve_", [])) + 1))}
    if hasattr(mlp, "loss_curve_"):
        hist["train_logloss"] = list(mlp.loss_curve_)
    if early_stopping and hasattr(mlp, "validation_scores_"):
        hist["internal_val_score"] = list(mlp.validation_scores_)
    if X_val is not None and y_val is not None:
        try:
            p_val = mlp.predict_proba(X_val)[:,1]
            hist["val_auc_once"] = float(roc_auc_score(y_val, p_val))
            hist["val_logloss_once"] = float(log_loss(y_val, p_val, labels=[0,1]))
        except Exception:
            pass

    stamp(f"[MLP] Trained with layers={hidden_layer_sizes}.")
    return mlp, hist

# ========= TabNet =========
def fit_tabnet_safe(
    X_train, y_train, X_val, y_val, *,
    X_test=None, y_test=None,
    width=24, n_steps=4, n_shared=2, n_independent=2, gamma=1.5,
    lambda_sparse=1e-5, mask_type="sparsemax",
    lr=6e-4, weight_decay=1e-4,
    max_epochs=120, patience=40,
    batch_size=256, virtual_batch_size=32,
    cat_idxs=None, cat_dims=None, cat_emb_dim=8,
    use_pretrain=False, pretrain_epochs=100, pretrain_patience=20, pretrain_bs=512,
    pretraining_ratio=0.8,
    random_state=0, use_gpu=True,
    es_min_delta=1e-4, es_monitor="val_0_auc", es_mode="max"
):
    """
    TabNet with robust OOM handling + BCEWithLogits(pos_weight) and history capture callback.
    """
    import numpy as np, warnings
    import torch
    import torch.nn as nn
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer
    from pytorch_tabnet.callbacks import Callback
    from sklearn.metrics import roc_auc_score, log_loss
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR

    def _to_float32(a):
        a = a.values if hasattr(a, "values") else a
        a = np.asarray(a, dtype=np.float32)
        a[~np.isfinite(a)] = 0.0
        return a

    def _to_int64(a):
        a = a.values if hasattr(a, "values") else a
        a = np.asarray(a)
        if a.dtype.kind not in "iu":
            a = a.astype(np.int64)
        else:
            a = a.astype(np.int64, copy=False)
        u = np.unique(a)
        if u.size == 2 and not np.array_equal(u, np.array([0, 1])):
            a = (a == u.max()).astype(np.int64)
        return a

    X_tr = _to_float32(X_train); X_va = _to_float32(X_val)
    y_tr = _to_int64(y_train);   y_va = _to_int64(y_val)

    sw = _balanced_sample_weight(y_tr)
    pos_w_scalar = float(_pos_weight(y_tr))

    want_cuda = bool(use_gpu)
    have_cuda = torch.cuda.is_available()
    device_pref = "cuda" if (want_cuda and have_cuda) else "cpu"

    # Early stopping callback (native or compat)
    try:
        from pytorch_tabnet.callbacks import EarlyStoppingCallback as _NativeES
        es_cb = _NativeES(patience=int(patience), min_delta=float(es_min_delta))
    except Exception:
        class _CompatEarlyStopping(Callback):
            def __init__(self, patience=10, min_delta=0.0, monitor="val_0_auc", mode="max"):
                super().__init__()
                self.patience=int(patience); self.min_delta=float(min_delta)
                self.monitor=monitor; self.mode=mode; self.best=None; self.wait=0
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}; val = logs.get(self.monitor)
                if val is None: return
                if self.best is None: self.best=val; self.wait=0; return
                improved = (val > self.best + self.min_delta) if self.mode=="max" else (val < self.best - self.min_delta)
                if improved: self.best=val; self.wait=0
                else:
                    self.wait+=1
                    if self.wait >= self.patience:
                        try: self.trainer._stop_training=True
                        except Exception: pass
        es_cb = _CompatEarlyStopping(patience=int(patience), min_delta=float(es_min_delta),
                                     monitor=str(es_monitor), mode=str(es_mode))

    class _CaptureHist(Callback):
        def __init__(self):
            super().__init__()
            self.buf = {"loss": [], "val_0_auc": [], "val_0_logloss": []}
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for k in list(self.buf.keys()):
                if k in logs and logs[k] is not None:
                    self.buf[k].append(float(logs[k]))

    cap_cb = _CaptureHist()

    cat_idxs = [] if cat_idxs is None else list(cat_idxs)
    cat_dims = [] if cat_dims is None else list(cat_dims)
    if isinstance(cat_emb_dim, int):
        cat_emb_dim = [cat_emb_dim] * len(cat_dims)
    elif cat_emb_dim is None:
        cat_emb_dim = [1] * len(cat_dims)
    else:
        cat_emb_dim = list(cat_emb_dim)
    assert len(cat_idxs) == len(cat_dims) == len(cat_emb_dim), "cat metadata lengths must match"

    tabnet_common = dict(
        n_d=width, n_a=width, n_steps=n_steps, gamma=gamma,
        n_shared=n_shared, n_independent=n_independent,
        lambda_sparse=lambda_sparse, mask_type=mask_type,
        seed=random_state, verbose=1, device_name=device_pref,
        optimizer_fn=AdamW, optimizer_params=dict(lr=float(lr), weight_decay=float(weight_decay)),
        scheduler_fn=StepLR, scheduler_params=dict(step_size=20, gamma=0.5),
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
    )

    def _fix_vbs(bs, vbs):
        bs = int(bs); vbs = int(vbs)
        if vbs > bs: vbs = max(16, min(128, bs // 2 if bs >= 32 else 16))
        if bs % vbs != 0:
            for d in (16, 24, 32, 48, 64, 96, 128, 192, 256):
                if d <= bs and bs % d == 0: vbs = d; break
        return bs, vbs

    batch_size, virtual_batch_size = _fix_vbs(batch_size, virtual_batch_size)

    # Binary-Logit loss wrapper (TabNet outputs 2 logits)
    def _make_bce_loss(device_name: str):
        dev = "cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu"
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w_scalar, dtype=torch.float32, device=dev))
        def _loss_fn(y_pred, y_true):
            z = (y_pred[:, 1] - y_pred[:, 0]).unsqueeze(1)
            y = y_true.float().unsqueeze(1)
            return bce(z, y)
        return _loss_fn

    ce_loss = nn.CrossEntropyLoss()

    def _fit_classifier(bs, vbs, device_name, pretrain_model=None):
        tn = TabNetClassifier(**{**tabnet_common, "device_name": device_name})
        if pretrain_model is not None:
            tn.load_from_unsupervised(pretrain_model)
        try:
            loss_fn = _make_bce_loss(device_name)
            _loss_for_fit = loss_fn
        except RuntimeError:
            _loss_for_fit = ce_loss

        stamp(f"[TabNet] Fit on {device_name}: width={tn.n_d}, steps={tn.n_steps}, bs={bs}/{vbs}, X_tr={X_tr.shape}, X_va={X_va.shape}")
        tn.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)], eval_name=["val"],
            eval_metric=["auc", "logloss"],
            patience=int(patience), max_epochs=int(max_epochs),
            batch_size=int(bs), virtual_batch_size=int(vbs),
            num_workers=0, pin_memory=(device_name == "cuda"),
            weights=sw, callbacks=[es_cb, cap_cb],
            loss_fn=_loss_for_fit
        )
        return tn

    # optional unsupervised pretraining
    pretrainer = None
    if use_pretrain:
        try:
            pre_bs, pre_vbs = _fix_vbs(int(pretrain_bs), max(16, int(pretrain_bs // 8)))
            pretrainer = TabNetPretrainer(
                **{k: v for k, v in tabnet_common.items()
                   if k not in ["optimizer_fn","optimizer_params","scheduler_fn","scheduler_params",
                                "cat_idxs","cat_dims","cat_emb_dim"]},
                optimizer_fn=AdamW,
                optimizer_params=dict(lr=float(lr), weight_decay=float(weight_decay)),
                scheduler_fn=StepLR, scheduler_params=dict(step_size=20, gamma=0.5),
            )
            pretrainer.fit(
                X_tr, eval_set=[X_va], eval_name=["val"],
                max_epochs=int(pretrain_epochs), patience=int(pretrain_patience),
                batch_size=pre_bs, virtual_batch_size=pre_vbs,
                num_workers=0, pin_memory=(device_pref == "cuda"),
                pretraining_ratio=float(pretraining_ratio)
            )
            stamp("[TabNet] Pretraining completed.")
        except RuntimeError as e:
            if device_pref == "cuda" and "out of memory" in str(e).lower():
                warnings.warn("TabNet pretraining OOM; skipping pretrain.", RuntimeWarning)
                try: torch.cuda.empty_cache()
                except Exception: pass
                pretrainer = None
            else:
                raise

    # try with preferred device; back off if OOM
    try:
        model = _fit_classifier(batch_size, virtual_batch_size, device_pref, pretrainer)
    except RuntimeError as e1:
        msg1 = str(e1).lower()
        if device_pref == "cuda" and ("out of memory" in msg1 or "cuda error" in msg1):
            warnings.warn("TabNet CUDA OOM; retrying with smaller config.", RuntimeWarning)
            try: torch.cuda.empty_cache()
            except Exception: pass
            width2 = max(12, width // 2)
            tabnet_common["n_d"] = width2
            tabnet_common["n_a"] = width2
            bs2, vbs2 = _fix_vbs(max(128, batch_size // 2), max(16, virtual_batch_size // 2))
            try:
                model = _fit_classifier(bs2, vbs2, "cuda", pretrainer)
            except RuntimeError as e2:
                msg2 = str(e2).lower()
                if "out of memory" in msg2 or "cuda error" in msg2:
                    warnings.warn("TabNet still OOM after backoff → switching to CPU.", RuntimeWarning)
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    model = _fit_classifier(bs2, vbs2, "cpu", pretrainer)
                else:
                    raise
        else:
            raise

    # history (robust to versions)
    hist_raw = {}
    try:
        if hasattr(model, "history"):
            if isinstance(model.history, dict):
                hist_raw = dict(model.history)
            elif hasattr(model.history, "history"):
                hist_raw = dict(model.history.history)
            else:
                hist_raw = {k: list(model.history[k]) for k in ("loss", "val_auc", "val_logloss") if k in model.history}
    except Exception:
        hist_raw = {}

    # fallback to captured logs
    if not hist_raw or all(len(v)==0 for v in hist_raw.values()):
        hist_raw = {k: list(v) for k, v in cap_cb.buf.items() if len(v) > 0}

    def _alias(src, *names):
        for n in names:
            if n in src:
                return list(src[n])
        return None

    hist = {}
    maybe = _alias(hist_raw, "loss")
    if maybe is not None: hist["loss"] = maybe
    maybe = _alias(hist_raw, "val_auc", "val_0_auc", "valid_auc", "valid_0_auc")
    if maybe is not None: hist["val_auc"] = maybe
    maybe = _alias(hist_raw, "val_logloss", "val_0_logloss", "valid_logloss", "valid_0_logloss")
    if maybe is not None: hist["val_logloss"] = maybe

    max_len = max((len(v) for v in hist.values()), default=0)
    hist["x"] = list(range(1, max_len + 1))

    if X_test is not None and y_test is not None:
        try:
            p_test = model.predict_proba(_to_numpy(X_test))[:, 1]
            hist["test_auc_once"] = float(roc_auc_score(y_test, p_test))
            hist["test_logloss_once"] = float(log_loss(y_test, p_test, labels=[0, 1]))
        except Exception:
            pass

    final_dev = getattr(model, "device_name", "cpu")
    stamp(f"[TabNet] Final device: {final_dev} (n_d/n_a={model.n_d}, steps={model.n_steps})")

    return model, hist

# ========= Calibration & Ensemble =========
def calibrate_model_isotonic(model, X_val, y_val):
    cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    cal.fit(X_val, y_val)
    return cal

class AveragingEnsemble:
    """Simple averaging ensemble over predict_proba of provided models."""
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights
        self.classes_ = np.array([0,1], dtype=np.int32)

    def predict_proba(self, X):
        p = None
        for name, mdl in self.models.items():
            w = float(self.weights.get(name, 0.0))
            if w <= 0: continue
            pi = mdl.predict_proba(X)[:,1]
            p = (w * pi) if p is None else (p + w * pi)
        if p is None:
            raise RuntimeError("Ensemble has no positive weights.")
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(np.int32)

def _weights_from_val_auc(models: Dict[str, Any], keys: Iterable[str], X_val, y_val) -> Dict[str, float]:
    scores = {}
    for k in keys:
        try:
            p = models[k].predict_proba(X_val)[:,1]
            scores[k] = float(roc_auc_score(y_val, p))
        except Exception:
            continue
    if not scores:
        raise RuntimeError("No AUCs available to derive ensemble weights.")
    eps = 1e-9
    pos_scores = {k: max(0.0, v - 0.5) for k, v in scores.items()}
    s = sum(pos_scores.values())
    if s < eps:
        return {k: 1.0/len(scores) for k in scores}
    return {k: v / s for k, v in pos_scores.items()}

# ========= Threshold optimization =========
def optimize_threshold(
    model, X_val, y_val, *, metric: str | Callable[[np.ndarray, np.ndarray, float], float] = "f1"
) -> float:
    """
    Tune decision threshold using validation probabilities. Supports:
      - metric="f1"       → max F1
      - metric="youden"   → max (sensitivity + specificity - 1)
      - metric=<callable> → signature (y_true, y_prob, t) -> score
    Returns the chosen threshold in [0,1].
    """
    p = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)

    if callable(metric):
        scorer = metric  # user-provided: (y_true, y_prob, t) -> float
    elif str(metric).lower() == "youden":
        from sklearn.metrics import recall_score
        def scorer(y_true, y_prob, t):
            y_hat = (y_prob >= t).astype(int)
            tn = np.sum((y_true == 0) & (y_hat == 0))
            fp = np.sum((y_true == 0) & (y_hat == 1))
            spec = tn / (tn + fp + 1e-9)
            sens = recall_score(y_true, y_hat)
            return sens + spec - 1.0
    else:
        from sklearn.metrics import f1_score
        def scorer(y_true, y_prob, t):
            return f1_score(y_true, (y_prob >= t).astype(int))

    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        s = scorer(y_val, p, float(t))
        if s > best_score:
            best_score, best_t = s, float(t)
    stamp(f"[THRESH] best_t={best_t:.3f}, score={best_score:.4f}")
    return float(best_t)

# ========= Plotting helpers =========
def _maybe_show():
    try:
        import matplotlib
        _ = matplotlib.get_backend()
    except Exception:
        pass

def plot_metric_curves(histories: Dict[str, Any]):
    """
    Plot AUC/logloss curves for XGB and TabNet; MLP shows internal validation score.
    Falls back to final reference lines if no per-iteration curves exist.
    """
    for name, hist in histories.items():
        plt.figure()
        x = hist.get("x", [])
        has_any = False

        if name == "xgb":
            for key, series in hist.items():
                if key == "x":
                    continue
                if key.endswith("_auc") or key.endswith("_logloss"):
                    plt.plot(x[:len(series)], series, label=key)
                    has_any = True
            plt.title("XGBoost: metrics over boosting rounds")

        elif name == "tabnet":
            for k in ("loss", "val_logloss", "val_auc"):
                if k in hist and len(hist[k]) > 0:
                    plt.plot(hist["x"][:len(hist[k])], hist[k], label=k)
                    has_any = True
            if not has_any:
                if "val_logloss" in hist and len(hist["val_logloss"]) > 0:
                    plt.axhline(hist["val_logloss"][-1], linestyle="--",
                                label=f"Final Val logloss = {hist['val_logloss'][-1]:.4f}")
                    has_any = True
                if "test_logloss_once" in hist:
                    plt.axhline(hist["test_logloss_once"], linestyle=":",
                                label=f"Final Test logloss = {hist['test_logloss_once']:.4f}")
            plt.title("TABNET — Validation/Test Curves")

        elif name == "mlp":
            if "train_logloss" in hist:
                plt.plot(range(1, len(hist["train_logloss"]) + 1),
                         hist["train_logloss"], label="train loss")
                has_any = True
            if "internal_val_score" in hist:
                plt.plot(range(1, len(hist["internal_val_score"]) + 1),
                         hist["internal_val_score"], label="internal val score")
                has_any = True
            if "val_auc_once" in hist:
                plt.axhline(hist["val_auc_once"], linestyle="--",
                            label=f"val_auc_once={hist['val_auc_once']:.3f}")
            plt.title("MLP: training loss & internal validation")

        if has_any:
            plt.xlabel("Iteration / Epoch")
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No per-iteration metrics available",
                     ha="center", va="center")
        plt.tight_layout()
        _maybe_show()

def plot_train_val_loss(histories: Dict[str, Any]):
    """Focus on loss-like metrics across the three models (where available)."""
    for name, hist in histories.items():
        x = hist.get("x", [])
        plt.figure()
        has_any = False
        if name == "xgb":
            for key, series in hist.items():
                if key == "x": continue
                if key.endswith("_logloss"):
                    plt.plot(x[:len(series)], series, label=key)
                    has_any = True
            plt.title("XGBoost: logloss")
        elif name == "tabnet":
            for key in ("loss", "val_logloss"):
                if key in hist:
                    plt.plot(x[:len(hist[key])], hist[key], label=key)
                    has_any = True
            plt.title("TabNet: loss/logloss")
        elif name == "mlp":
            if "train_logloss" in hist:
                plt.plot(range(1, len(hist["train_logloss"])+1), hist["train_logloss"], label="train loss")
                has_any = True
            plt.title("MLP: training loss")
        if has_any:
            plt.xlabel("Iteration / Epoch"); plt.grid(True); plt.legend(); plt.tight_layout(); _maybe_show()
        else:
            plt.close()

def _roc_pr_one(model, X, y):
    p = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap  = average_precision_score(y, p)
    return fpr, tpr, prec, rec, auc, ap

def plot_roc_all(models: Dict[str, Any], X, y, split_name="valid"):
    plt.figure()
    for name, m in models.items():
        try:
            fpr, tpr, _, _, auc, _ = _roc_pr_one(m, X, y)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.title(f"ROC — {split_name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.grid(True); plt.legend(); plt.tight_layout()
    _maybe_show()

def plot_pr_all(models: Dict[str, Any], X, y, split_name="valid"):
    plt.figure()
    for name, m in models.items():
        try:
            _, _, prec, rec, _, ap = _roc_pr_one(m, X, y)
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            continue
    plt.title(f"Precision–Recall — {split_name}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True); plt.legend(); plt.tight_layout()
    _maybe_show()
