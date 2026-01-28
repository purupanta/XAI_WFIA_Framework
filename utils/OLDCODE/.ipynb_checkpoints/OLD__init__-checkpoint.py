# utils/__init__.py

# --- Silence TensorFlow & Python warnings (must come before TF imports) ---
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=INFO+WARN, 3=ERROR only
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Namespace imports (safe, no side effects)
from . import _config
from . import helper
from . import applib
from . import _0_catboost
from . import _1_3_train_valid_test_split
from . import _2_ml_rf
from . import _2_ml_lr

# Optional: re-export selected symbols
from .helper import *
from ._config import *
from .applib import *
from ._0_catboost import *
from ._1_3_train_valid_test_split import *
from ._2_ml_rf import *
from ._2_ml_lr import *

VERSION = "1.0"

def _safe_all(module):
    return getattr(module, "__all__", [name for name in dir(module) if not name.startswith("_")])

__all__ = (
    _safe_all(_config)
    + _safe_all(helper)
    + _safe_all(applib)
    + _safe_all(_0_catboost)
    + _safe_all(_1_3_train_valid_test_split)
    + _safe_all(_2_ml_rf)
    + _safe_all(_2_ml_lr)
    + ["VERSION"]
)

# Keep init quiet — no prints, no device allocations here


print(">>>Utils Initialized<<<")