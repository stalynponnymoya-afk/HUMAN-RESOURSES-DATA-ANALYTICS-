"""Data loading, standardization and imputation for the Smart Cargo dataset.

The original notebook hard-coded the cargo labels used for salary imputation
(``"asesor_comercial"``, ``"auxiliar_de_servicios_en_general"`` ...). Here the
imputation is driven by the observed data instead: the mode of a cargo is only
used when that cargo has enough observations, and everything else falls back to
the group (company) average. That keeps the script working when the dataset —
or the wording of a job title — changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Loading / persistence
# ---------------------------------------------------------------------------
def load_dataset(path: PathLike, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset, dispatching on file extension.

    Supports ``.xlsx``/``.xls`` (via openpyxl) and ``.csv``. The extension
    check keeps the loader usable outside of Google Colab where the data now
    lives on disk under ``data/raw``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en '{path}'. "
            "Colócalo en data/raw/ o pasa una ruta explícita."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        # sheet_name=None makes pandas return a dict of every sheet, so only
        # forward it when the caller actually asked for a specific sheet.
        if sheet_name is None:
            return pd.read_excel(path)
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato de archivo no soportado: '{suffix}'")


def save_processed(df: pd.DataFrame, path: PathLike) -> Path:
    """Persist the cleaned dataset to disk (defaults to ``data/processed``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------
def standardize_categoricals(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Lower-case and underscore-separate every categorical/object column.

    Works on a copy so the caller's frame is untouched, and skips missing
    values instead of turning them into the literal ``"nan"``.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].apply(
            lambda x: x.strip().lower().replace(" ", "_") if isinstance(x, str) else x
        )
    return df


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------
def _mode_or_none(series: pd.Series) -> Optional[float]:
    """Return the mode of ``series`` or ``None`` when it carries no data."""
    series = series.dropna()
    if series.empty:
        return None
    return series.mode().iloc[0]


def impute_salary_by_cargo(
    df: pd.DataFrame,
    salary_col: str,
    cargo_col: str,
    min_observations: int = 1,
) -> pd.DataFrame:
    """Fill missing salaries with the modal salary of each cargo.

    A cargo is only used as a reference when it has at least
    ``min_observations`` non-null salaries; otherwise it is left for the
    company-level fallback. This replaces the previous hard-coded lookup of
    two specific job titles.
    """
    df = df.copy()
    cargo_stats = (
        df.dropna(subset=[salary_col])
        .groupby(cargo_col)[salary_col]
        .agg(["count", "mean"])
    )
    cargo_stats = cargo_stats[cargo_stats["count"] >= min_observations]

    modes = (
        df.dropna(subset=[salary_col])
        .groupby(cargo_col)[salary_col]
        .agg(lambda s: s.mode().iloc[0])
    )

    mask = df[salary_col].isnull() & df[cargo_col].isin(modes.index)
    df.loc[mask, salary_col] = df.loc[mask, cargo_col].map(modes)
    return df


def impute_salary_by_director(
    df: pd.DataFrame,
    salary_col: str,
    cargo_col: str,
    director_pattern: str = "director",
) -> pd.DataFrame:
    """Fill missing director salaries with the mean director salary."""
    df = df.copy()
    is_director = df[cargo_col].str.contains(director_pattern, case=False, na=False)
    mean_salary = df.loc[is_director, salary_col].mean()
    if pd.isna(mean_salary):
        return df
    mask = is_director & df[salary_col].isnull()
    df.loc[mask, salary_col] = mean_salary
    return df


def impute_salary_by_group(
    df: pd.DataFrame, salary_col: str, group_col: str
) -> pd.DataFrame:
    """Fallback: fill remaining missing salaries with the group mean."""
    df = df.copy()
    group_means = df.groupby(group_col)[salary_col].transform("mean")
    df[salary_col] = df[salary_col].fillna(group_means)
    return df


def impute_categorical_by_group(
    df: pd.DataFrame, column: str, group_col: str
) -> pd.DataFrame:
    """Fill a missing categorical column with the group's modal value."""
    df = df.copy()
    for group_value, group in df.groupby(group_col):
        if group[column].isna().any():
            mode = _mode_or_none(group[column])
            if mode is not None:
                df.loc[
                    (df[group_col] == group_value) & df[column].isna(), column
                ] = mode
    return df


def impute_numeric_by_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill a missing numeric column with its global mean."""
    df = df.copy()
    if column in df.columns and df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mean())
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean_dataset(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the tidy frame.

    Order matters: salaries are imputed cargo-first (most specific), then by
    director role, then by company as a last resort; ``area`` and ``edad`` are
    filled afterwards.
    """
    df = standardize_categoricals(df, config)

    df = impute_salary_by_cargo(
        df, config.salary_col, config.cargo_col, config.min_salary_observations
    )
    df = impute_salary_by_director(
        df, config.salary_col, config.cargo_col, config.director_pattern
    )
    df = impute_salary_by_group(df, config.salary_col, config.group_col)

    df = impute_categorical_by_group(df, config.area_col, config.group_col)
    df = impute_numeric_by_mean(df, config.age_col)

    return df


def load_and_clean(
    path: PathLike,
    config: PipelineConfig = DEFAULT_CONFIG,
    save: bool = True,
    output_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Convenience wrapper: load raw data, clean it, optionally persist it."""
    df = load_dataset(path)
    df = clean_dataset(df, config)
    if save:
        from .config import DEFAULT_PROCESSED_PATH

        save_processed(df, output_path or DEFAULT_PROCESSED_PATH)
    return df


def missing_value_report(df: pd.DataFrame) -> Dict[str, int]:
    """Return the count of missing values per column (non-zero entries)."""
    missing = df.isnull().sum()
    return missing[missing > 0].to_dict()
