# utils/base_configs.py

__all__ = [
    "base_configs",
    "base_config_dict",
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_BASE_CONFIG_DICT",
    "DEFAULT_BASE_CONFIG_JSON",
]

from dataclasses import dataclass, asdict
from typing import Tuple
from datetime import datetime
import os

# Decide device once at import time
try:
    import torch
    _DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"

# --- One stable timestamp for the whole run (set at module import time)
_RUN_TS: str = datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass(frozen=True)
class base_configs:
    # Paths / columns
    IP_CSV_HINTS6_CLEANED_ENCODED: str = "op/3_cleanedEncoded/hints6_public_filtered_v1_cleaned_encoded.csv"
    IP_CSV_HINTS6_7_CLEANED_ENCODED: str = "ip/3_cleanedEncoded/hints6_7_cleaned_encoded.csv"
    TARGET_COL_NAME: str = "MedConditions_HeartCondition"

    # Train-Val-Test split
    DIR_tr_va_te_split_SAVE_DIR: str = "op/tr_va_te_split_"+_RUN_TS+"/"
    DIR_tr_va_te_split_SAVE_PREFIX: str = ""        # e.g., target col to prefix files
    DIR_tr_va_te_split_SAVE_FORMAT: str = "csv"     # 'parquet' | 'csv'

    # Train-Val-Test metrics and SHAP
    DIR_tr_va_te_metric_shap_SAVE_DIR: str = "op/tr_va_te_metric_shap_"+_RUN_TS+"/"

    # Train-Val-Test plot
    DIR_tr_va_te_plot_SAVE_DIR: str = "op/tr_va_te_plot_"+_RUN_TS+"/"

    
    # Hardware
    DEVICE: str = _DEVICE
    USE_GPU: bool = (_DEVICE == "cuda")

    # Repro
    RANDOM_STATE: int = 42

    # Model sizes / epochs
    RF_TREES: int = 400
    XGB_TREES: int = 500
    TABNET_EPOCHS: int = 40
    TABNET_PATIENCE: int = 8
    TABNET_WIDTH: int = 24
    MLP_ITERS: int = 120
    MLP_HIDDEN: Tuple[int, int] = (96, 48)

    # IO
    IN_DIR: str = "ip"
    OUT_DIR: str = "op"

     # --- Expose the timestamp inside the config so it appears in asdict/json
    RUN_TS: str = _RUN_TS


def get_base_configs(return_type=None, verbose=0):
    cfg = asdict(base_configs())
    out = None
    if return_type == "json":
        import json
        out = json.dumps(cfg, indent=2)
    elif return_type == "dict" or return_type is None:
        out = cfg
    else:
        raise ValueError("return_type must be None, 'dict', or 'json'")
    if verbose > 0:
        print("Config Info:", out)
    return out

