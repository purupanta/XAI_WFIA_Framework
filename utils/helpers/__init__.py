# utils/helpers/__init__.py

from __future__ import annotations

from . import (
    dir_helpers                as _dir_helpers,
    logging_helpers            as _logging_helpers,
    rw_csv_helpers             as _rw_csv_helpers,
    feature_distr_helpers      as _feature_distr_helpers,
    feature_transform_helpers  as _feature_transform_helpers,
    tr_va_te_eval_helpers      as _tr_va_te_eval_helpers,
    tr_va_te_result_plot       as _tr_va_te_result_plot,
)

def _public(mod):
    # use __all__ if present, else export non-private names
    return list(getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")]))

_modules = (
    _dir_helpers,
    _logging_helpers,
    _rw_csv_helpers,
    _feature_distr_helpers,
    _feature_transform_helpers,
    _tr_va_te_eval_helpers,
    _tr_va_te_result_plot,
)

# hoist public names
for _m in _modules:
    for _n in _public(_m):
        globals()[_n] = getattr(_m, _n)

# export list
__all__ = sorted({name for _m in _modules for name in _public(_m)})
