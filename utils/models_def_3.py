# utils/models_def.py

from utils.helpers import logging_helpers

__all__ = [
    "train_models",
    "tune_all_models",
    "predict_proba_ensemble",
    "fit_lr_safe",
    "fit_rf_safe",
    "fit_xgb_safe",
    "fit_mlp_safe",
    "fit_tabnet_safe",
]

# ========= Safe fit helpers & utilities =========

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # xgboost optional
    XGBClassifier = None


def stamp(msg):  # lightweight logger
    print(msg)


# ---- Shared: balanced sample weights ----
def _balanced_sample_weight(y):
    import numpy as np
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=np.float32)
    w_pos = 0.5 / max(pos, 1)
    w_neg = 0.5 / max(neg, 1)
    sw = np.where(y == 1, w_pos, w_neg).astype(np.float32)
    return sw * y.size  # normalize ~N


# ----- LR -----
def fit_lr_safe(
    X, y, *, 
    C=1.0, penalty="l2", l1_ratio=None,
    max_iter=400, class_weight="balanced",
    random_state=0, prefer_saga=True
):
    solvers = (["saga", "lbfgs", "liblinear"] if prefer_saga else ["lbfgs", "saga", "liblinear"])
    last_err = None
    for solver in solvers:
        # Only solvers that support the chosen penalty
        if penalty == "elasticnet" and solver != "saga":
            continue
        if penalty in ("l1", "elasticnet") and solver == "lbfgs":
            continue

        params = dict(
            C=float(C), penalty=penalty, l1_ratio=l1_ratio,
            max_iter=max_iter, class_weight=class_weight,
            random_state=random_state, solver=solver,
            n_jobs=-1 if solver == "saga" else None,
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                lr = LogisticRegression(**{k: v for k, v in params.items() if v is not None})
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


# ----- RF -----
def fit_rf_safe(
    X, y, *,
    n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced_subsample", max_samples=None,
    random_state=0
):
    def _try(n_trees, depth):
        rf = RandomForestClassifier(
            n_estimators=int(n_trees), max_depth=depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            class_weight=class_weight, n_jobs=-1, random_state=random_state,
            bootstrap=True, oob_score=False, max_samples=max_samples
        )
        rf.fit(X, y)
        return rf
    try:
        rf = _try(n_estimators, max_depth)
        stamp(f"[RF] Trained with {n_estimators} trees, depth={max_depth}, leaf={min_samples_leaf}.")
        return rf
    except Exception as e:
        stamp(f"[RF] Error; retrying smaller. Reason: {e}")
    n_small = max(100, int(n_estimators) // 2)
    d_small = 20 if (max_depth is None or (isinstance(max_depth, int) and max_depth > 20)) else max_depth
    rf = _try(n_small, d_small)
    stamp(f"[RF] Using fallback {n_small} trees, depth={d_small}.")
    return rf


# ===== Pure-CPU XGBoost (no QuantileDMatrix, no CUDA) =====
class _XGBPureCPUWrapper:
    """Minimal scikit-style wrapper around a Booster."""
    def __init__(self, booster, n_features_):
        import numpy as np
        self._booster = booster
        self.n_features_in_ = int(n_features_)
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def predict_proba(self, X):
        import numpy as np, xgboost as xgb
        X = np.asarray(X, dtype=np.float32)
        dm = xgb.DMatrix(X)
        p1 = self._booster.predict(dm)  # prob of class 1
        p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        import numpy as np
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)

    def get_xgb_params(self):
        return dict(self._booster.attributes())


def _pos_weight(y):
    import numpy as np
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return (neg / max(pos, 1)) if pos > 0 else 1.0


def fit_xgb_pure_cpu(
    X, y, *, n_estimators=1200, learning_rate=0.04, max_depth=5,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0,
    random_state=0, eval_metric="logloss"
):
    import numpy as np, xgboost as xgb
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int32)

    dm_train = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": eval_metric,
        "tree_method": "hist",
        "predictor": "cpu_predictor",
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "lambda": float(reg_lambda),
        "seed": int(random_state),
    }
    booster = xgb.train(
        params=params,
        dtrain=dm_train,
        num_boost_round=int(n_estimators),
        verbose_eval=False
    )
    model = _XGBPureCPUWrapper(booster, n_features_=X.shape[1])
    stamp("[XGB] Trained pure CPU Booster.")
    return model


def fit_xgb_safe(
    X, y, X_val=None, y_val=None, *,
    n_estimators=1200, learning_rate=0.04, max_depth=5,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0,
    min_child_weight=1, gamma=0.0,
    random_state=0
):
    # Prefer sklearn wrapper when available to get early_stopping
    if XGBClassifier is not None:
        spw = _pos_weight(y)
        try:
            xgb = XGBClassifier(
                objective="binary:logistic",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=0.0,
                min_child_weight=min_child_weight,
                gamma=gamma,
                eval_metric="logloss",
                tree_method="hist",
                predictor="cpu_predictor",
                random_state=random_state,
                n_jobs=-1,
                scale_pos_weight=spw,
            )
            if X_val is not None and y_val is not None:
                xgb.fit(X, y, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)
            else:
                xgb.fit(X, y)
            stamp("[XGB] Trained sklearn wrapper (CPU, hist) with early stopping." if X_val is not None else "[XGB] Trained sklearn wrapper (CPU, hist).")
            return xgb
        except Exception as e:
            stamp(f"[XGB] sklearn wrapper failed; using pure CPU fallback. Reason: {e}")

    # Fallback to pure CPU trainer (no early stopping)
    return fit_xgb_pure_cpu(
        X, y, n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, random_state=random_state, eval_metric="logloss"
    )


# ----- MLP (auto-handle missing sample_weight support) -----
def _shrink_hidden(hsizes):
    # e.g., (128,64) -> (64,32) -> (32,) -> (16,) -> (8,)
    if isinstance(hsizes, int):
        hsizes = (hsizes,)
    hs = list(hsizes)
    if len(hs) > 1:
        hs = [max(8, h // 2) for h in hs]
        if hs[-1] < 16 and len(hs) > 1:
            hs = hs[:-1]
    else:
        hs[0] = max(8, hs[0] // 2)
    return tuple(hs)


def fit_mlp_safe(
    X, y, *,
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
    alpha=5e-4, learning_rate_init=8e-4, max_iter=250,
    early_stopping=True, n_iter_no_change=20, class_weight="balanced",
    random_state=0
):
    import warnings, inspect

    # compute weights if requested
    sample_weight = None
    if class_weight == "balanced":
        sample_weight = _balanced_sample_weight(y)

    # check whether this sklearn supports sample_weight
    def _supports_sample_weight():
        try:
            sig = inspect.signature(MLPClassifier.fit)
            return "sample_weight" in sig.parameters
        except Exception:
            return False

    sw_ok = _supports_sample_weight()

    def _try(hsizes, iters):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            mlp = MLPClassifier(
                hidden_layer_sizes=hsizes, activation=activation, solver=solver,
                alpha=alpha, learning_rate_init=learning_rate_init, max_iter=iters,
                early_stopping=early_stopping, n_iter_no_change=n_iter_no_change,
                random_state=random_state
            )
            if sw_ok and sample_weight is not None:
                mlp.fit(X, y, sample_weight=sample_weight)
            else:
                if (sample_weight is not None) and (not sw_ok):
                    stamp("[MLP] sample_weight not supported by this sklearn; training without it.")
                mlp.fit(X, y)
            if any(isinstance(wi.message, ConvergenceWarning) for wi in w):
                stamp(f"[MLP] ConvergenceWarning; increasing max_iter to {iters*2}.")
                if sw_ok and sample_weight is not None:
                    mlp.set_params(max_iter=iters*2, warm_start=True).fit(X, y, sample_weight=sample_weight)
                else:
                    mlp.set_params(max_iter=iters*2, warm_start=True).fit(X, y)
            return mlp

    try:
        mlp = _try(hidden_layer_sizes, max_iter)
        stamp(f"[MLP] Trained with layers={hidden_layer_sizes}.")
        return mlp
    except Exception as e:
        stamp(f"[MLP] Error; shrinking network. Reason: {e}")
        small = _shrink_hidden(hidden_layer_sizes)
        mlp = _try(small, max_iter)
        stamp(f"[MLP] Using fallback layers={small}.")
        return mlp


# ===========================
# TabNet (improved & corrected) with Early Stopping (compat)
# ===========================

def fit_tabnet_safe(
    X_train, y_train, X_val, y_val, *,
    # model capacity
    width=24, n_steps=4, n_shared=2, n_independent=2, gamma=1.5,
    lambda_sparse=1e-6, mask_type="entmax",  # or "sparsemax"
    # optimization
    lr=1e-3, weight_decay=1e-4,
    max_epochs=300, patience=40,
    batch_size=256, virtual_batch_size=32,
    # categorical metadata (optional)
    cat_idxs=None, cat_dims=None, cat_emb_dim=8,
    # pretraining (optional)
    use_pretrain=False, pretrain_epochs=200, pretrain_patience=30, pretrain_bs=512,
    pretraining_ratio=0.8,
    # device control
    random_state=0, use_gpu=True,
    # early stopping tuning
    es_min_delta=5e-4, es_monitor="val_0_auc", es_mode="max"
):
    """
    Train TabNet with best-practice defaults for tabular classification.
    Includes Early Stopping with min_delta that works across TabNet versions.
    """
    import numpy as np, warnings
    import torch
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        from pytorch_tabnet.pretraining import TabNetPretrainer
        from pytorch_tabnet.callbacks import Callback
    except Exception as e:
        raise ImportError("pytorch_tabnet is required for TabNet. Please install pytorch-tabnet.") from e

    # Try to import native ES; otherwise define a compat callback.
    try:
        from pytorch_tabnet.callbacks import EarlyStoppingCallback as _NativeES
        def make_es_cb():
            return _NativeES(patience=int(patience), min_delta=float(es_min_delta))
    except Exception:
        class _CompatEarlyStopping(Callback):
            def __init__(self, patience=10, min_delta=0.0, monitor="val_0_auc", mode="max"):
                super().__init__()
                self.patience = int(patience)
                self.min_delta = float(min_delta)
                self.monitor = monitor
                self.mode = mode
                self.best = None
                self.wait = 0
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                key = str(self.monitor)
                val = logs.get(key, logs.get("valid_0_auc", logs.get("val_0_auc")))
                if val is None:
                    return
                if self.best is None:
                    self.best = val; self.wait = 0; return
                improved = (val > self.best + self.min_delta) if self.mode == "max" else (val < self.best - self.min_delta)
                if improved:
                    self.best = val; self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        try: self.trainer._stop_training = True
                        except Exception: pass
        def make_es_cb():
            return _CompatEarlyStopping(
                patience=int(patience),
                min_delta=float(es_min_delta),
                monitor=str(es_monitor),
                mode=str(es_mode)
            )

    want_cuda = bool(use_gpu)
    have_cuda = torch.cuda.is_available()
    device = "cuda" if (want_cuda and have_cuda) else "cpu"
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ---------- Sanitize inputs ----------
    def _to_float32(a):
        a = a.values if hasattr(a, "values") else a
        import numpy as _np
        a = _np.asarray(a, dtype=_np.float32)
        a[~_np.isfinite(a)] = 0.0
        return a

    def _to_int64(a):
        a = a.values if hasattr(a, "values") else a
        import numpy as _np
        a = _np.asarray(a)
        if a.dtype.kind not in "iu":
            a = a.astype(_np.int64)
        else:
            a = a.astype(_np.int64, copy=False)
        u = _np.unique(a)
        if u.size == 2 and not _np.array_equal(u, _np.array([0, 1])):
            a = (a == u.max()).astype(_np.int64)
        return a

    X_tr = _to_float32(X_train); X_va = _to_float32(X_val)
    y_tr = _to_int64(y_train);   y_va = _to_int64(y_val)

    # sample weights
    sw = _balanced_sample_weight(y_tr)

    # virtual batch must divide batch_size
    def _fix_vbs(bs, vbs):
        bs = int(bs); vbs = int(vbs)
        if vbs > bs:
            vbs = max(16, min(128, bs // 2 if bs >= 32 else 16))
        if bs % vbs != 0:
            for d in (16, 24, 32, 48, 64, 96, 128, 192, 256):
                if d <= bs and bs % d == 0:
                    vbs = d; break
        return int(bs), int(vbs)

    batch_size, virtual_batch_size = _fix_vbs(batch_size, virtual_batch_size)

    # ---------- Normalize categorical args ----------
    if cat_idxs is None: cat_idxs = []
    if cat_dims is None: cat_dims = []
    cat_idxs = list(cat_idxs); cat_dims = list(cat_dims)
    if isinstance(cat_emb_dim, int):
        cat_emb_dim = [cat_emb_dim] * len(cat_dims)
    elif cat_emb_dim is None:
        cat_emb_dim = [1] * len(cat_dims)
    else:
        cat_emb_dim = list(cat_emb_dim)
    assert len(cat_idxs) == len(cat_dims) == len(cat_emb_dim), (
        f"cat_idxs, cat_dims, cat_emb_dim must have same length; "
        f"got {len(cat_idxs)}, {len(cat_dims)}, {len(cat_emb_dim)}"
    )

    # ---------- Optimizer & scheduler (StepLR: metric-agnostic) ----------
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR

    tabnet_common = dict(
        n_d=width, n_a=width, n_steps=n_steps, gamma=gamma,
        n_shared=n_shared, n_independent=n_independent,
        lambda_sparse=lambda_sparse, mask_type=mask_type,
        seed=random_state, verbose=1,              # per-epoch logs for visibility
        device_name=device,

        optimizer_fn=AdamW,
        optimizer_params=dict(lr=float(lr), weight_decay=float(weight_decay)),
        scheduler_fn=StepLR,
        scheduler_params=dict(step_size=20, gamma=0.5),

        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
    )

    def _fit_classifier(pretrain_model=None, bs=None, vbs=None, w=None):
        tn = TabNetClassifier(**tabnet_common)
        if pretrain_model is not None:
            tn.load_from_unsupervised(pretrain_model)

        bs = int(bs if bs is not None else batch_size)
        vbs = int(vbs if vbs is not None else virtual_batch_size)

        stamp(f"[TabNet] About to fit: bs={bs}, vbs={vbs}, "
              f"X_tr={X_tr.shape}, X_va={X_va.shape}, device={device}")

        es_cb = make_es_cb()

        tn.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)], eval_name=["val"],
            eval_metric=["auc"], patience=int(patience), max_epochs=int(max_epochs),
            batch_size=bs, virtual_batch_size=vbs,
            num_workers=0, pin_memory=(device == "cuda"),
            weights=(w if w is not None else sw),
            callbacks=[es_cb]
        )

        # Optional diagnostics
        try:
            hist = tn.history
            auc_hist = hist.get('val_0_auc', hist.get('valid_0_auc', []))
            if len(auc_hist):
                import numpy as np
                best_epoch = int(np.argmax(auc_hist))
                best_auc = float(np.max(auc_hist))
                stamp(f"[TabNet] best_epoch={best_epoch}, best_val_auc={best_auc:.5f}, "
                      f"epochs_ran={len(auc_hist)}")
        except Exception:
            pass

        stamp("[TabNet] fit() returned OK")
        return tn

    # ---------- Optional unsupervised pretraining ----------
    pretrainer = None
    if use_pretrain:
        try:
            pre_bs, pre_vbs = _fix_vbs(int(pretrain_bs), max(16, int(pretrain_bs // 8)))
            pretrainer = TabNetPretrainer(
                **{k: v for k, v in tabnet_common.items()
                   if k not in ["cat_idxs", "cat_dims", "cat_emb_dim",
                                "optimizer_fn", "optimizer_params",
                                "scheduler_fn", "scheduler_params"]},
                optimizer_fn=AdamW,
                optimizer_params=dict(lr=float(lr), weight_decay=float(weight_decay)),
                scheduler_fn=StepLR,
                scheduler_params=dict(step_size=20, gamma=0.5),
            )
            pretrainer.fit(
                X_tr, eval_set=[X_va], eval_name=["val"],
                max_epochs=int(pretrain_epochs), patience=int(pretrain_patience),
                batch_size=pre_bs, virtual_batch_size=pre_vbs,
                num_workers=0, pin_memory=(device == "cuda"),
                pretraining_ratio=float(pretraining_ratio)
            )
        except RuntimeError as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                warnings.warn("TabNet pretraining OOM on GPU; skipping pretrain.", RuntimeWarning)
                import torch as _torch
                _torch.cuda.empty_cache()
                pretrainer = None
            else:
                raise

    # ---------- Fit with minimal OOM backoff ----------
    try:
        model = _fit_classifier(pretrainer)
        if device == "cuda":
            from torch.cuda import get_device_name
            dev = torch.cuda.current_device()
            stamp(f"[TabNet] GPU on {get_device_name(dev)} "
                  f"(width={width}, steps={n_steps}, bs={batch_size}/{virtual_batch_size}, lr={lr})")
        else:
            stamp(f"[TabNet] CPU (width={width}, steps={n_steps}, bs={batch_size}/{virtual_batch_size}, lr={lr})")
        return model

    except RuntimeError as e:
        msg = str(e)
        if device == "cuda" and "out of memory" in msg.lower():
            import torch as _torch
            import warnings as _warnings
            _warnings.warn("TabNet CUDA OOM; retrying once with a smaller config.", RuntimeWarning)
            _torch.cuda.empty_cache()
            # back off once
            width2 = max(12, width // 2)
            bs2, vbs2 = _fix_vbs(max(128, batch_size // 2), max(16, virtual_batch_size // 2))
            tabnet_common.update(n_d=width2, n_a=width2)
            try:
                model = _fit_classifier(pretrainer, bs=bs2, vbs=vbs2)
                from torch.cuda import get_device_name
                dev = _torch.cuda.current_device()
                stamp(f"[TabNet] GPU (backoff) on {get_device_name(dev)} "
                      f"(width={width2}, steps={n_steps}, bs={bs2}/{vbs2}, lr={lr})")
                return model
            except RuntimeError:
                _warnings.warn("TabNet still OOM after backoff → switching to CPU.", RuntimeWarning)
                _torch.cuda.empty_cache()
                tabnet_common.update(device_name="cpu")
                model = _fit_classifier(pretrainer, bs=bs2, vbs=vbs2)
                stamp("[TabNet] Fallback → CPU (backoff config).")
                return model
        else:
            raise


# ========= Simple ensemble utility =========

def predict_proba_ensemble(models: dict, X, weights: dict | None = None):
    """
    Average probabilities across any subset of models present in `models`.
    weights: optional dict like {"xgb": 0.5, "lr": 0.3, "tabnet": 0.2}
    """
    import numpy as np
    if not models:
        raise ValueError("No models provided for ensembling.")
    if weights is None:
        weights = {k: 1.0 for k in models.keys()}
    # Normalize weights over models that exist
    keys = [k for k in weights.keys() if k in models]
    w = np.array([weights[k] for k in keys], dtype=float)
    if not len(keys):
        raise ValueError("No overlapping model names between models and weights.")
    w = w / w.sum()

    preds = []
    for k in keys:
        m = models[k]
        # TabNet and sklearn models share predict_proba
        p = m.predict_proba(X)[:, 1]
        preds.append(p)
    P = np.stack(preds, axis=0)  # [M, N]
    return (w[:, None] * P).sum(axis=0)  # [N]


# ========= Lightweight calibration (optional) =========

def _calibrate_if_requested(model, X_cal, y_cal, method="isotonic", cv="prefit"):
    """
    Calibrates model probabilities using sklearn's CalibratedClassifierCV.
    Returns the calibrated model (wrapper) or the original if calibration fails.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        cal = CalibratedClassifierCV(model, method=method, cv=cv)
        cal.fit(X_cal, y_cal)
        stamp(f"[CAL] Applied {method} calibration.")
        return cal
    except Exception as e:
        stamp(f"[CAL] Skipping calibration (reason: {e}).")
        return model


# ========= Tuning scaffold (grids crafted for strong default lifts) =========

def tune_all_models(
    X_tr_s, y_tr,      # scaled for LR/MLP
    X_tr,    X_val, y_val,   # unscaled for RF/XGB/TabNet
    random_state=0
):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def _auc(y_true, p): return float(roc_auc_score(y_true, p))
    def _eval_proba(model): return _auc(y_val, model.predict_proba(X_val)[:, 1])

    results = {}

    # ---- LR grid ----
    lr_grid = [
        dict(C=0.3, penalty="l2"),
        dict(C=1.0, penalty="l2"),
        dict(C=3.0, penalty="l2"),
        dict(C=0.5, penalty="elasticnet", l1_ratio=0.1),
        dict(C=1.0, penalty="elasticnet", l1_ratio=0.3),
    ]
    best_score, best_model = -np.inf, None
    for g in lr_grid:
        m = fit_lr_safe(X_tr_s, y_tr, random_state=random_state, prefer_saga=True, **g)
        s = _eval_proba(m)
        if s > best_score: best_score, best_model = s, m
    results["lr"] = (best_model, best_score)

    # ---- RF grid ----
    rf_grid = [
        dict(n_estimators=800, max_depth=12, min_samples_leaf=1, max_features="sqrt"),
        dict(n_estimators=1000, max_depth=16, min_samples_leaf=2, max_features=0.5),
        dict(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features="sqrt"),
    ]
    best_score, best_model = -np.inf, None
    for g in rf_grid:
        m = fit_rf_safe(X_tr, y_tr, random_state=random_state, class_weight="balanced_subsample", **g)
        s = _eval_proba(m)
        if s > best_score: best_score, best_model = s, m
    results["rf"] = (best_model, best_score)

    # ---- XGB grid ----
    xgb_grid = [
        dict(learning_rate=0.03, max_depth=5, n_estimators=1200, subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0),
        dict(learning_rate=0.05, max_depth=4, n_estimators=800,  subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0),
        dict(learning_rate=0.02, max_depth=6, n_estimators=1500, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0),
    ]
    best_score, best_model = -np.inf, None
    for g in xgb_grid:
        m = fit_xgb_safe(X_tr, y_tr, X_val=X_val, y_val=y_val, random_state=random_state, **g)
        s = _eval_proba(m)
        if s > best_score: best_score, best_model = s, m
    results["xgb"] = (best_model, best_score)

    # ---- MLP grid ----
    mlp_grid = [
        dict(hidden_layer_sizes=(64, 32),  alpha=1e-4, learning_rate_init=8e-4),
        dict(hidden_layer_sizes=(128, 64), alpha=5e-4, learning_rate_init=1e-3),
        dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=5e-4),
    ]
    best_score, best_model = -np.inf, None
    for g in mlp_grid:
        m = fit_mlp_safe(X_tr_s, y_tr, random_state=random_state, **g)
        s = _eval_proba(m)
        if s > best_score: best_score, best_model = s, m
    results["mlp"] = (best_model, best_score)

    # ---- TabNet grid ----
    tn_grid = [
        dict(width=24, n_steps=4, lr=1e-3,  gamma=1.5, weight_decay=1e-4, batch_size=256,  virtual_batch_size=32),
        dict(width=32, n_steps=4, lr=8e-4,  gamma=1.5, weight_decay=1e-4, batch_size=512,  virtual_batch_size=64),
        dict(width=48, n_steps=5, lr=1e-3,  gamma=1.0, weight_decay=5e-4, batch_size=512,  virtual_batch_size=64),
    ]
    best_score, best_model = -np.inf, None
    for g in tn_grid:
        m = fit_tabnet_safe(X_tr, y_tr, X_val, y_val,
                            patience=20, max_epochs=300, use_gpu=True,
                            es_min_delta=5e-4, es_monitor="val_0_auc", es_mode="max",
                            cat_idxs=[], cat_dims=[], cat_emb_dim=[],
                            **g)
        s = _eval_proba(m)
        if s > best_score: best_score, best_model = s, m
    results["tabnet"] = (best_model, best_score)

    models = {name: model for name, (model, _) in results.items()}
    leaderboard = {name: float(score) for name, (_, score) in results.items()}
    return models, leaderboard


# ========= MAIN CALLING FUNCTION (with tuning, ensemble, calibration) =========

def train_models(
    CONFIGS,
    X_train_res_scaled, y_train_res,
    X_train_res, X_val, y_val,
    models=None
):
    """
    Trains LR, RF, XGB, MLP, TabNet.
    If CONFIGS['TUNE'] is True, runs a small tuned search and returns best variants.
    If CONFIGS['CALIBRATE'] is True, calibrates each model on (X_val, y_val).
    If CONFIGS['ENSEMBLE'] is True, adds 'ensemble' entry with averaged probabilities helper.
    """
    if models is None:
        models = {}

    TUNE       = bool(CONFIGS.get("TUNE", False))
    ENSEMBLE   = bool(CONFIGS.get("ENSEMBLE", False))
    CALIBRATE  = bool(CONFIGS.get("CALIBRATE", False))
    CALIB_METHOD = str(CONFIGS.get("CALIB_METHOD", "isotonic"))

    if TUNE:
        with logging_helpers.Timer("Tune & Train (all models)"):
            tuned_models, leaderboard = tune_all_models(
                X_train_res_scaled, y_train_res,
                X_train_res, X_val, y_val,
                random_state=CONFIGS['RANDOM_STATE']
            )
            models.update(tuned_models)
            stamp(f"[TUNE] Leaderboard (val ROC AUC): {leaderboard}")
    else:
        with logging_helpers.Timer("Train Logistic Regression"):
            models["lr"] = fit_lr_safe(
                X_train_res_scaled, y_train_res,
                max_iter=400, class_weight="balanced",
                random_state=CONFIGS['RANDOM_STATE'], prefer_saga=True
            )

        with logging_helpers.Timer("Train Random Forest"):
            models["rf"] = fit_rf_safe(
                X_train_res, y_train_res,
                n_estimators=CONFIGS.get('RF_TREES', 800), max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                max_features="sqrt", class_weight="balanced_subsample",
                random_state=CONFIGS['RANDOM_STATE']
            )

        with logging_helpers.Timer("Train XGBoost"):
            models["xgb"] = fit_xgb_safe(
                X_train_res, y_train_res,
                X_val=X_val, y_val=y_val,
                random_state=CONFIGS['RANDOM_STATE'],
                n_estimators=1200, learning_rate=0.04, max_depth=5,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0
            )

        with logging_helpers.Timer("Train MLP"):
            models["mlp"] = fit_mlp_safe(
                X_train_res_scaled, y_train_res,
                hidden_layer_sizes=CONFIGS.get('MLP_HIDDEN', (128, 64)),
                random_state=CONFIGS['RANDOM_STATE']
            )

        with logging_helpers.Timer("Train TabNet"):
            models["tabnet"] = fit_tabnet_safe(
                X_train_res, y_train_res,
                X_val, y_val,
                width=24, n_steps=4,
                max_epochs=300, patience=20,
                batch_size=256, virtual_batch_size=32,
                lr=1e-3, gamma=1.5, weight_decay=1e-4,
                random_state=CONFIGS['RANDOM_STATE'], use_gpu=True,
                cat_idxs=[], cat_dims=[], cat_emb_dim=[],
                es_min_delta=5e-4, es_monitor="val_0_auc", es_mode="max"
            )

    # Optional calibration on validation set
    if CALIBRATE:
        with logging_helpers.Timer(f"Calibrate models ({CALIB_METHOD})"):
            for name in list(models.keys()):
                try:
                    # TabNet & sklearn share predict_proba; wrap with CalibratedClassifierCV
                    models[name] = _calibrate_if_requested(models[name], X_val, y_val, method=CALIB_METHOD)
                except Exception as e:
                    stamp(f"[CAL] {name}: skip ({e})")

    # Optional ensemble entry (no model object; you call predict_proba_ensemble)
    if ENSEMBLE:
        # nothing to train here; we expose an entry for clarity
        models["__ensemble__"] = "use predict_proba_ensemble(models, X, weights)"

    return models
