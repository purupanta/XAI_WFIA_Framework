# utils/helpers/dir_helpers.py

__all__ = [
    "get_cwd"
]

def get_cwd(verbose=0):
    from pathlib import Path
    cwd = Path.cwd()
    if(verbose > 0):
        print("Current working directory:", cwd)
    return cwd
