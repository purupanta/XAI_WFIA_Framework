__all__ = [
    "data_preprocessing",
    "print_preprocessing_report"
]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

_LINE  = "-" * 50
_BREAK = "=" * 50

# --------- section printers (same look as before) ---------
def _class_table(y: pd.Series) -> pd.DataFrame:
    vc  = y.value_counts(dropna=False).sort_index()
    pct = (vc / vc.sum() * 100).round(2)
    tbl = pd.DataFrame({"count": vc.astype(int), "pct%": pct})
    tbl.index.name = y.name or "target"
    return tbl

def _print_original(X, y, verbose: int):
    print("1) Original data:")
    print("   Complete dataset:\n")
    print(f"a. Complete dataset: X{X.shape}, y={len(y)}")
    if verbose >= 2:
        print(_class_table(y))
    print("\n" + _BREAK + "\n")

def _print_split_block(X_train, y_train, X_val, y_val, X_test, y_test, *, y_full_len: int, verbose: int):
    pt = len(y_train) / y_full_len * 100
    pv = len(y_val)   / y_full_len * 100
    pte= len(y_test)  / y_full_len * 100
    print("2) After Train/Validation/Test split:")
    print(f"(Distribution: Train = {len(y_train)} ({pt:.2f}%) + Validation = {len(y_val)} ({pv:.2f} %) + Test = {len(y_test)} ({pte:.2f}%))")
    print(_LINE)
    print(f"(2.1) Train: X{X_train.shape}, y={len(y_train)}")
    if verbose >= 2: print(_class_table(y_train)); print(_LINE)
    print(f"(2.2) Validation: X{X_val.shape}, y={len(y_val)}")
    if verbose >= 2: print(_class_table(y_val));   print(_LINE)
    print(f"(2.3) Test: X{X_test.shape}, y={len(y_test)}")
    if verbose >= 2: print(_class_table(y_test));  print("\n" + _BREAK + "\n")

def _print_after_balance(balance_method, X_train_res, y_train_res, X_val, y_val, X_test, y_test, verbose: int):
    print(f"3) After train-only balance ({balance_method}):")
    print(_LINE)
    print(f"(3.1) Train: X{X_train_res.shape}, y={len(y_train_res)}")
    if verbose >= 2: print(_class_table(y_train_res)); print(_LINE)
    print(f"(3.2) Validation: X{X_val.shape}, y={len(y_val)}")
    if verbose >= 2: print(_class_table(y_val));       print(_LINE)
    print(f"(3.3) Test: X{X_test.shape}, y={len(y_test)}")
    if verbose >= 2: print(_class_table(y_test));      print("\n" + _BREAK + "\n")

def _print_after_scale(X_train_res_scaled, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test, verbose: int):
    print("4) After train-only scale (standard scale):")
    print(_LINE)
    print(f"(4.1) Train: X{X_train_res_scaled.shape}, y={len(y_train_res)}")
    if verbose >= 2: print(_class_table(y_train_res)); print(_LINE)
    print(f"(4.2) Validation: X{X_val_scaled.shape}, y={len(y_val)}")
    if verbose >= 2: print(_class_table(y_val));       print(_LINE)
    print(f"(4.3) Test: X{X_test_scaled.shape}, y={len(y_test)}")
    if verbose >= 2: print(_class_table(y_test));      print(_BREAK)

# --------- single public printer you asked for ---------
def print_preprocessing_report(
    verbose: int,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series,
    X_train_res: pd.DataFrame, y_train_res: pd.Series,
    X_train_res_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    balance_method: str
):
    if verbose <= 0:
        return

    # (1) Original
    _print_original(X, y, verbose)

    # (2) Split
    _print_split_block(
        X_train, y_train, X_val, y_val, X_test, y_test,
        y_full_len=len(y), verbose=verbose
    )

    # (3) Balance training only
    _print_after_balance(
        balance_method, X_train_res, y_train_res, X_val, y_val, X_test, y_test, verbose
    )

    # (4) Scale (fit on balanced train)
    _print_after_scale(
        X_train_res_scaled, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test, verbose
    )
    return "Successfully printed tables"


# =========================
# A) Split (60/20/20)
# =========================
def split_60_20_20(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    stratify: bool = True,
):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test) with ~60/20/20."""
    strat_vec = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=strat_vec, random_state=random_state
    )
    strat_vec_temp = y_temp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=strat_vec_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# =========================
# B) Balance (train only)
# =========================
def balance_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str = "adasyn",  # 'adasyn' | 'smote' | 'smoteenn' | 'none'
    random_state: int = 42,
    **kwargs,
):
    """Resample training only; returns X_train_res, y_train_res. Lazy-imports imblearn."""
    if method == "none":
        return X_train.copy(), y_train.copy()
    try:
        if method == "adasyn":
            from imblearn.over_sampling import ADASYN
            balancer = ADASYN(random_state=random_state, **kwargs)
        elif method == "smote":
            from imblearn.over_sampling import SMOTE
            balancer = SMOTE(random_state=random_state, **kwargs)
        elif method == "smoteenn":
            from imblearn.combine import SMOTEENN
            balancer = SMOTEENN(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
    except ImportError:
        print("[warn] imbalanced-learn not installed; proceeding without resampling.")
        return X_train.copy(), y_train.copy()

    Xr, yr = balancer.fit_resample(X_train, y_train)
    return pd.DataFrame(Xr, columns=X_train.columns), pd.Series(yr, name=y_train.name)

# =========================
# C) Scale (fit on balanced train)
# =========================
def scale_train_val_test(
    X_train_res: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    scaler: StandardScaler | None = None
):
    """Fit StandardScaler on balanced train; transform val/test. Returns X_tr_s, X_val_s, X_te_s, scaler."""
    scaler = scaler or StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X_train_res.columns, index=X_train_res.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val),          columns=X_val.columns,       index=X_val.index)
    X_te_s  = pd.DataFrame(scaler.transform(X_test),         columns=X_test.columns,      index=X_test.index)
    return X_tr_s, X_val_s, X_te_s, scaler

# =========================
# D) Orchestrator (thin)
# =========================
def data_preprocessing(
    verbose: int,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    stratify: bool = True,
    balance_method: str = "adasyn",     # 'adasyn' | 'smote' | 'smoteenn' | 'none'
    balance_kwargs: dict | None = None
):
    """Split → balance(train) → scale, with formatted prints when verbose>=1."""
    balance_kwargs = balance_kwargs or {}

    # (1) Original
    # We originally have this table already
    
    # (2) Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(
        X, y, random_state=random_state, stratify=stratify
    )
    # (3) Balance training only
    X_train_res, y_train_res = balance_train_only(
        X_train, y_train, method=balance_method, random_state=random_state, **balance_kwargs
    )
    # (4) Scale (fit on balanced train)
    X_train_res_scaled, X_val_scaled, X_test_scaled, scaler = scale_train_val_test(
        X_train_res, X_val, X_test
    )

    result = {
        # raw splits
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,     "y_val": y_val,
        "X_test": X_test,   "y_test": y_test,

        # balanced train
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,

        # scaled (fit on balanced train)
        "X_train_res_scaled": X_train_res_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "features": X.columns.tolist(),
        "scaler": scaler,
    }

    if verbose > 0:
        # After you run your pipeline and have `result`,
        # The following code prints and displays the table
        print_msg = print_preprocessing_report(
            verbose=2,
            X=X, y=y,
            X_train=result["X_train"], y_train=result["y_train"],
            X_val=result["X_val"],     y_val=result["y_val"],
            X_test=result["X_test"],   y_test=result["y_test"],
            X_train_res=result["X_train_res"], y_train_res=result["y_train_res"],
            X_train_res_scaled=result["X_train_res_scaled"],
            X_val_scaled=result["X_val_scaled"],
            X_test_scaled=result["X_test_scaled"],
            balance_method="adasyn"  # or 'smote' | 'smoteenn' | 'none'
        )

    return result
