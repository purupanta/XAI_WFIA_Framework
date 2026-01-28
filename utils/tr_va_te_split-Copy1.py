# utils/tr_va_te_split.py

# =============================================================
# Preprocessing Orchestrator (complete module)
# =============================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional imbalance handlers (install: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.combine import SMOTEENN
except Exception:  # pragma: no cover
    ADASYN = SMOTE = SMOTEENN = None


# =========================
# Utilities
# =========================
def _class_counts(y: pd.Series) -> dict:
    vc = pd.Series(y).value_counts(dropna=False).sort_index()
    # coerce keys to int if possible
    out = {}
    for k, v in vc.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            out[str(k)] = int(v)
    return out

def _pct(n: int, d: int) -> str:
    return f"{n} ({(100.0*n/d):.1f}%)" if d else f"{n} (n/a%)"

def _save_df(df: pd.DataFrame | pd.Series, path: Path, fmt: str = "parquet"):
    """Save DataFrame/Series to disk as parquet or csv (suffix auto-added)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name or "target")
    fmt = fmt.lower()
    if fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"))
    elif fmt == "csv":
        df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        raise ValueError("save_format must be 'parquet' or 'csv'")

# =========================
# Core steps
# =========================
def split_60_20_20(
    X: pd.DataFrame, y: pd.Series,
    *, random_state: int = 42, stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X_train, X_val, X_test, y_train, y_val, y_test) with 60/20/20."""
    strat = y if stratify else None

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=strat
    )

    # Split remaining 40% equally into val/test (20/20 overall)
    strat_tmp = y_tmp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state, stratify=strat_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str = "adasyn",  # 'adasyn' | 'smote' | 'smoteenn' | 'none'
    random_state: int = 42,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance the training set only. Returns (X_res, y_res)."""
    method = (method or "none").lower()
    if method == "none":
        return X_train, y_train

    if ADASYN is None or SMOTE is None or SMOTEENN is None:
        raise ImportError(
            "imbalanced-learn is required for balancing. Install with: pip install imbalanced-learn"
        )

    if method == "adasyn":
        sampler = ADASYN(random_state=random_state, **kwargs)
    elif method == "smote":
        sampler = SMOTE(random_state=random_state, **kwargs)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=random_state, **kwargs)
    else:
        raise ValueError("balance method must be one of: 'adasyn', 'smote', 'smoteenn', 'none'")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    # Ensure DataFrame/Series types with column/index names preserved when possible
    X_res = pd.DataFrame(X_res, columns=getattr(X_train, "columns", None))
    y_res = pd.Series(y_res, name=getattr(y_train, "name", None))
    return X_res, y_res


def scale_train_val_test(
    X_train_res: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    scaler: StandardScaler | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler on balanced train features, transform train/val/test.
    Returns scaled DataFrames and the fitted scaler.
    """
    scaler = scaler or StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_res),
        columns=X_train_res.columns, index=X_train_res.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# =========================
# Reporting
# =========================
def print_preprocessing_report(
    *,
    verbose: int,           # 1 basic, 2 detailed, 3 full
    X: pd.DataFrame, y: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series,
    X_train_res: pd.DataFrame, y_train_res: pd.Series,
    X_train_res_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    balance_method: str = "adasyn",
) -> None:
    """Layered report; keeps output compact at low verbosity, expands at higher levels."""
    # ---- Tier 1: Basic ----
    if verbose >= 1:
        print("────────────────────────────────────────────────────────")
        print("Preprocessing report (basic)")
        print(f"• Features: {X.shape[1]} | Target: {y.name!r}")
        print(f"• Total rows: {len(X):,}")
        print(f"• Split sizes 60/20/20 → "
              f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"• y distribution (orig): {_class_counts(y)}")

    # ---- Tier 2: Detailed ----
    if verbose >= 2:
        print("\nBalancing (train only)")
        b, a = _class_counts(y_train), _class_counts(y_train_res)
        tb, ta = sum(b.values()), sum(a.values())
        print(f"• Method: {balance_method}")
        for cls in sorted(set(b) | set(a)):
            print(f"  - class {cls}: {_pct(b.get(cls, 0), tb)} → {_pct(a.get(cls, 0), ta)}")

        print("\nScaling (fit on balanced train)")
        print(f"• X_train_res_scaled shape: {X_train_res_scaled.shape}")
        print(f"• X_val_scaled shape:       {X_val_scaled.shape}")
        print(f"• X_test_scaled shape:      {X_test_scaled.shape}")

    # ---- Tier 3: Full ----
    if verbose >= 3:
        print("\nAdditional diagnostics")
        # NA columns
        na_cols = X.columns[X.isna().any()].tolist()
        if na_cols:
            preview = ", ".join(map(str, na_cols[:12]))
            print(f"• Columns with NA in original X: {len(na_cols)} → {preview}{' ...' if len(na_cols)>12 else ''}")
        # basic scaling stats
        try:
            mu = np.asarray(X_train_res_scaled.mean(axis=0)).ravel()
            sd = np.asarray(X_train_res_scaled.std(axis=0)).ravel()
            print(f"• Scaled means μ: mean={mu.mean():.3f} ± {mu.std():.3f}")
            print(f"• Scaled stds  σ: mean={sd.mean():.3f} ± {sd.std():.3f}")
        except Exception:
            pass
        # per-split distributions
        print(f"• y_train: {_class_counts(y_train)}")
        print(f"• y_val:   {_class_counts(y_val)}")
        print(f"• y_test:  {_class_counts(y_test)}")
        print("────────────────────────────────────────────────────────")

# =========================
# Orchestrator
# =========================
def data_preprocessing(
    CONFIGS, 
    verbose: int,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    stratify: bool = True,
    balance_method: str = "adasyn",   # 'adasyn' | 'smote' | 'smoteenn' | 'none'
    balance_kwargs: dict | None = None,
    balance_val_for_diagnostics: bool = True,  # produce X_val_bal / y_val_bal for inspection only
) -> Dict[str, Any]:
    """
    Split → balance(train) → scale(fit on balanced train).

    Verbosity tiers:
      <=0: mute
       >0: basic info report
       >1: basic + detailed
       >2: basic + detailed + full diagnostics

    Saving behavior is controlled by CONFIGS at top of file.
    """
    balance_kwargs = balance_kwargs or {}

    # ---------------- (1) Split ----------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(
        X, y, random_state=random_state, stratify=stratify
    )

    if verbose > 0:
        print("▸ Split 60/20/20 complete.")
        print(f"  - X shape: {X.shape} | features: {X.shape[1]}")
        print(f"  - y name: {y.name!r} | classes: {_class_counts(y)}")
        print(f"  - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if verbose > 2:
        na_cols = X.columns[X.isna().any()].tolist()
        if na_cols:
            print(f"  - Columns with NA in X (orig): {len(na_cols)}")
        print(f"  - y distribution by split → "
              f"train={_class_counts(y_train)}, val={_class_counts(y_val)}, test={_class_counts(y_test)}")

    # ---------------- (2) Balance training only ----------------
    X_train_res, y_train_res = balance_train_only(
        X_train, y_train, method=balance_method, random_state=random_state, **balance_kwargs
    )

    if verbose > 1:
        before = _class_counts(y_train)
        after  = _class_counts(y_train_res)
        tb, ta = sum(before.values()), sum(after.values())
        print(f"▸ Balancing (train) with '{balance_method}'")
        for k in sorted(set(before) | set(after)):
            b = before.get(k, 0); a = after.get(k, 0)
            print(f"    class {k}: {_pct(b, tb)} → {_pct(a, ta)}")

    # ---------------- (3) Optional: balance validation (diagnostics only) -----
    if balance_val_for_diagnostics and balance_method != "none":
        X_val_bal, y_val_bal = balance_train_only(
            X_val, y_val, method=balance_method, random_state=random_state, **balance_kwargs
        )
        if verbose > 2:
            print("▸ Validation (diagnostic) balanced (not used for selection).")
    else:
        X_val_bal, y_val_bal = None, None

    # ---------------- (4) Scale (fit on balanced train) -----------------------
    X_train_res_scaled, X_val_scaled, X_test_scaled, scaler = scale_train_val_test(
        X_train_res, X_val, X_test
    )

    if verbose > 1:
        print(f"▸ Scaling with {type(scaler).__name__} (fit on balanced train).")

    # ---------------- (5) Bundle result --------------------------------------
    result: Dict[str, Any] = {
        # raw splits
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,

        # balanced train
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,

        # scaled (fit on balanced train)
        "X_train_res_scaled": X_train_res_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        # diagnostics (optional)
        "X_val_bal": X_val_bal,
        "y_val_bal": y_val_bal,

        "features": X.columns.tolist(),
        "scaler": scaler,
    }

    # ---------------- (6) Structured report ----------------------------------
    if verbose > 0:
        report_level = 1 if verbose == 1 else (2 if verbose == 2 else 3)
        print_preprocessing_report(
            verbose=report_level,
            X=X, y=y,
            X_train=result["X_train"], y_train=result["y_train"],
            X_val=result["X_val"],     y_val=result["y_val"],
            X_test=result["X_test"],   y_test=result["y_test"],
            X_train_res=result["X_train_res"], y_train_res=result["y_train_res"],
            X_train_res_scaled=result["X_train_res_scaled"],
            X_val_scaled=result["X_val_scaled"],
            X_test_scaled=result["X_test_scaled"],
            balance_method=balance_method,
        )
        if verbose > 2 and X_val_bal is not None:
            print("▸ (Extra) Diagnostic balanced val distribution:", _class_counts(y_val_bal))

    # ---------------- (7) Save the artifacts to the disk ---------------------------------- 

    result["save_dir"] = save_tr_va_te_artifacts(
        CONFIGS,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_train_res=X_train_res, y_train_res=y_train_res,
        X_train_res_scaled=X_train_res_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        X_val_bal=X_val_bal, y_val_bal=y_val_bal,
        verbose=verbose,
    )

    return result


from pathlib import Path
import joblib

def save_tr_va_te_artifacts(
    CONFIGS,
    *,
    X_train, y_train, X_val, y_val, X_test, y_test,
    X_train_res, y_train_res,
    X_train_res_scaled, X_val_scaled, X_test_scaled,
    scaler,
    X_val_bal=None, y_val_bal=None,
    _save_df=_save_df,  # keep your existing saver
    verbose: int = 0,
):
    """Persist splits/artifacts based on CONFIGS; returns save_dir or None."""
    save_dir_cfg    = CONFIGS.get("DIR_tr_va_te_split_SAVE_DIR", None)
    save_prefix_cfg = CONFIGS.get("DIR_tr_va_te_split_SAVE_PREFIX", "")
    save_format_cfg = CONFIGS.get("DIR_tr_va_te_split_SAVE_FORMAT", "parquet")

    save_dir = Path(save_dir_cfg) if save_dir_cfg not in (None, "", False) else None
    if save_dir is None:
        return None

    sp = (save_prefix_cfg.strip() + "_") if save_prefix_cfg else ""
    save_dir.mkdir(parents=True, exist_ok=True)

    # raw splits
    _save_df(X_train, save_dir / f"{sp}X_train", save_format_cfg)
    _save_df(y_train, save_dir / f"{sp}y_train", save_format_cfg)
    _save_df(X_val,   save_dir / f"{sp}X_val",   save_format_cfg)
    _save_df(y_val,   save_dir / f"{sp}y_val",   save_format_cfg)
    _save_df(X_test,  save_dir / f"{sp}X_test",  save_format_cfg)
    _save_df(y_test,  save_dir / f"{sp}y_test",  save_format_cfg)

    # balanced train
    _save_df(X_train_res, save_dir / f"{sp}X_train_res", save_format_cfg)
    _save_df(y_train_res, save_dir / f"{sp}y_train_res", save_format_cfg)

    # scaled
    _save_df(X_train_res_scaled, save_dir / f"{sp}X_train_res_scaled", save_format_cfg)
    _save_df(X_val_scaled,       save_dir / f"{sp}X_val_scaled",       save_format_cfg)
    _save_df(X_test_scaled,      save_dir / f"{sp}X_test_scaled",      save_format_cfg)

    # validation balanced (optional diagnostic)
    if X_val_bal is not None:
        _save_df(X_val_bal, save_dir / f"{sp}X_val_bal", save_format_cfg)
    if y_val_bal is not None:
        _save_df(y_val_bal, save_dir / f"{sp}y_val_bal", save_format_cfg)

    # scaler artifact
    art_dir = save_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, art_dir / f"{sp}scaler.joblib")

    if verbose > 0:
        print(f"▸ Artifacts saved to: {save_dir.resolve()} (format={save_format_cfg})")

    return save_dir


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Synthetic demo (remove in production)
    rng = np.random.default_rng(0)
    n = 1000
    X_demo = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(loc=2.0, scale=0.5, size=n),
        "f3": rng.integers(0, 5, size=n)
    })
    y_demo = pd.Series((rng.random(n) < 0.15).astype(int), name="target")

    # Configure saving (optional)
    CONFIGS["DIR_tr_va_te_split_SAVE_DIR"] = None          # set a folder to enable saving
    CONFIGS["DIR_tr_va_te_split_SAVE_PREFIX"] = "target"   # e.g., your real target col
    CONFIGS["DIR_tr_va_te_split_SAVE_FORMAT"] = "parquet"  # or "csv"

    result = data_preprocessing(
        verbose=3,
        X=X_demo, y=y_demo,
        balance_method="adasyn",
        balance_kwargs={"n_neighbors": 5},
        balance_val_for_diagnostics=True,
    )
