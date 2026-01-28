# utils/helpers/rw_csv_helpers.py

__all__ = [
    "read_csv_file",
    "write_csv_file"
]

from pathlib import Path
import pandas as pd

def read_csv_file(csv_path, verbose=0, columns=None, **read_csv_kwargs):
    """
    Read a CSV at `path` and optionally print a tiny preview.

    Args:
        path (str | Path): File path (relative or absolute).
        verbose (int): 0=silent; 1=path+shape; 2=+sample preview.
        n_sample (int): Rows to show when verbose>1.
        **read_csv_kwargs: Passed to pandas.read_csv (optional).
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Safer default: read all at once to avoid chunk-based type inference warnings
    read_csv_kwargs.setdefault("low_memory", False)

    # map `columns` to pandas `usecols`
    if columns is not None:
        read_csv_kwargs.setdefault("usecols", columns)

    df = pd.read_csv(csv_path, **read_csv_kwargs)

    if verbose > 0:
        print(f"Loaded: {csv_path}")
        print("─" * 80)
        print(f"Shape: {df.shape}")
        print("─" * 80)
        print(f"All columns: {list(df.columns)}")

    if verbose > 1:
        n_sample = 3 # First three rows to show
        print("─" * 80)
        print(df.head(n_sample))
        
    return df


def write_csv_file(path, df, verbose=0, n_sample=5, overwrite=True, index=False, **to_csv_kwargs):
    """
    Write DataFrame `df` to CSV at `path`.

    Args:
        path (str | Path): Output file path (relative or absolute).
        df (pd.DataFrame): DataFrame to save.
        verbose (int): 0=silent; 1=path+shape; 2=+sample preview.
        n_sample (int): Rows to show when verbose>1.
        overwrite (bool): If False and file exists -> error.
        index (bool): Save index column to CSV.
        **to_csv_kwargs: Passed to DataFrame.to_csv (optional).
    Returns:
        str: Absolute path written.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")

    csv_file_path = Path(path).resolve()
    if csv_file_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {csv_file_path}")

    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file_path, index=index, **to_csv_kwargs)

    if verbose > 0:
        print(f"Saved: {csv_file_path}\nshape: {df.shape}")
    if verbose > 1:
        k = max(0, min(n_sample, len(df)))
        print(df.sample(k, random_state=0) if k > 0 else "[empty dataframe]")

    return str(csv_file_path)
    