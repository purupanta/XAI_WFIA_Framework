# utils/__init__.py

from __future__ import annotations
VERSION = "1.0"

from . import deps as _deps
from . import base_configs as _base_configs
from . import catboost_lib as _catboost_lib
from . import models_def as _models_def
from .import pipeline_lr as _pipeline_lr
from .import pipeline_rf as _pipeline_rf
from .import pipeline_xgb as _pipeline_xgb
from .import pipeline_xlstm as _pipeline_xlstm
from .import pipeline_tabnet as _pipeline_tabnet
from .import pipeline_mlp as _pipeline_mlp
from .import pipeline_node as _pipeline_node

from .import shap_utils as _shap_utils
from .import shap_explain as _shap_explain

# 1) import the helpers subpackage
from . import helpers as _helpers   # this is utils/helpers/__init__.py

def _public(mod):
    return list(getattr(mod, "__all__", []))

# 2) re-export helpers' public names into utils.*
for _n in _public(_helpers):
    globals()[_n] = getattr(_helpers, _n)

# 3) build __all__
__all__ = sorted(set(
    _public(_helpers)
    + _public(_deps)
    + _public(_base_configs)
    + _public(_catboost_lib)
    + _public(_models_def)
    + ["VERSION"]
))
