# utils/models_def.py

from utils.helpers import logging_helpers

__all__ = [
    "train_models",
    "fit_lr_safe",
    "fit_rf_safe",
    "fit_xgb_safe",
    "fit_mlp_safe",
    "fit_tabnet_safe",
]

# ========= MAIN CALLING FUNCTION =========

def train_models(
    CONFIGS,
    X_train_res_scaled, y_train_res,
    X_train_res, X_val, y_val,
    models=None
):
    if models is None:
        models = {}

    with logging_helpers.Timer("Train Logistic Regression"):
        models["lr"] = fit_lr_safe(
            X_train_res_scaled, y_train_res,
            max_iter=400, class_weight="balanced",
            random_state=CONFIGS['RANDOM_STATE'], prefer_saga=True
        )

    with logging_helpers.Timer("Train Random Forest"):
        models["rf"] = fit_rf_safe(
            X_train_res, y_train_res,
            n_estimators=CONFIGS['RF_TREES'], max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=CONFIGS['RANDOM_STATE']
        )

    with logging_helpers.Timer("Train XGBoost"):
        models["xgb"] = fit_xgb_safe(
            X_train_res, y_train_res,
            X_val=X_val, y_val=y_val,
            random_state=CONFIGS['RANDOM_STATE']
        )

    with logging_helpers.Timer("Train MLP"):
        models["mlp"] = fit_mlp_safe(
            X_train_res_scaled, y_train_res,
            hidden_layer_sizes=CONFIGS['MLP_HIDDEN'],
            random_state=CONFIGS['RANDOM_STATE']
        )

    with logging_helpers.Timer("Train TabNet"):
        models["tabnet"] = fit_tabnet_safe(
            X_train_res, y_train_res,
            X_val, y_val,
            width=12, n_steps=3,
            max_epochs=200, patience=10,
            batch_size=256, virtual_batch_size=32,
            lr=1e-3, gamma=1.5,
            random_state=CONFIGS['RANDOM_STATE'], use_gpu=True,  # True or False
            cat_idxs=[], cat_dims=[], cat_emb_dim=[],
            es_min_delta=5e-4, es_monitor="val_0_auc", es_mode="max"
        )

    return models


# ========= END OF MAIN CALLING FUNCTION =========


# ========= Safe fit helpers =========
# models_def.py
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


# ----- Shared: balanced sample weights -----
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
    n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced_subsample", max_samples=None,
    random_state=0
):
    def _try(n_trees, depth):
        rf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=depth, min_samples_split=min_samples_split,
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
    n_small = max(100, n_estimators // 2)
    d_small = 20 if (max_depth is None or max_depth > 20) else max_depth
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

    # for compatibility if you log params
    def get_xgb_params(self):
        return dict(self._booster.attributes())


def _pos_weight(y):
    import numpy as np
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return (neg / max(pos, 1)) if pos > 0 else 1.0


def fit_xgb_pure_cpu(
    X, y, *, n_estimators=1500, learning_rate=0.03, max_depth=5,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
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
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
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
    alpha=2e-4, learning_rate_init=8e-4, max_iter=200,
    early_stopping=True, n_iter_no_change=10, class_weight="balanced",
    random_state=0
):
    import warnings, inspect
    from sklearn.exceptions import ConvergenceWarning

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
    width=24, n_steps=3, n_shared=2, n_independent=2, gamma=1.5,
    lambda_sparse=1e-6, mask_type="entmax",  # "sparsemax" also fine
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
    es_min_delta=1e-4,           # minimum improvement that resets patience
    es_monitor="val_0_auc",      # metric key in logs/history
    es_mode="max"                # "max" for AUC/accuracy, "min" for loss
):
    """
    Train TabNet with best-practice defaults for tabular classification.
    Includes Early Stopping with min_delta that works across TabNet versions.
    """
    import numpy as np, warnings
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer

    # Try to import native ES; otherwise define a compat callback.
    try:
        from pytorch_tabnet.callbacks import EarlyStoppingCallback as _NativeES
        def make_es_cb():
            return _NativeES(patience=int(patience), min_delta=float(es_min_delta))
    except Exception:
        from pytorch_tabnet.callbacks import Callback
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
                # try a few common metric keys for safety
                val = logs.get(key)
                if val is None:
                    val = logs.get("valid_0_auc", logs.get("val_0_auc"))
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

    # ---------- Device ----------
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
        # force binary {0,1} if it's e.g. {0,2} or {-1,1}
        if u.size == 2 and not np.array_equal(u, np.array([0, 1])):
            a = (a == u.max()).astype(np.int64)
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
    cat_idxs = list(cat_idxs)
    cat_dims = list(cat_dims)
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
                torch.cuda.empty_cache()
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
            import math
            warnings.warn("TabNet CUDA OOM; retrying once with a smaller config.", RuntimeWarning)
            import torch as _torch
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
                # fall back to CPU
                warnings.warn("TabNet still OOM after backoff → switching to CPU.", RuntimeWarning)
                _torch.cuda.empty_cache()
                tabnet_common.update(device_name="cpu")
                model = _fit_classifier(pretrainer, bs=bs2, vbs=vbs2)
                stamp("[TabNet] Fallback → CPU (backoff config).")
                return model
        else:
            raise

# ========= End helpers =========
