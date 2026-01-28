# utils/catboost_lib.py

__all__ = [
    "impute_negatives_gpu_catboost_fast"
]


import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from catboost import CatBoostRegressor

def impute_negatives_gpu_catboost_fast(
    df: pd.DataFrame,
    max_iter=4,
    random_state=42,
    depth=6,
    iterations=150,
    learning_rate=0.08,
    n_nearest_features=20,
    use_gpu=True
) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].mask(df[c] < 0, np.nan)

    est = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=random_state,
        task_type="GPU" if use_gpu else "CPU",
        devices="0",
        gpu_ram_part=0.6,
        allow_writing_files=False,
        verbose=False
    )

    imp = IterativeImputer(
        estimator=est,
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy="median",
        imputation_order="ascending",
        n_nearest_features=n_nearest_features,
        skip_complete=True,
        tol=1e-3
    )

    df[num_cols] = imp.fit_transform(df[num_cols].astype(np.float32))
    return df
