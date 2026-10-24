"""Feature engineering, encoding, scaling and splitting (Phase 2).

Turns the cleaned frame into model-ready matrices: derives the ratio/indicator
features, separates low- from high-cardinality categoricals (one-hot vs target
encoding), scales numerics, balances the target with SMOTE and performs the
stratified train/val/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import DEFAULT_CONFIG, PipelineConfig


@dataclass
class PreparedData:
    """Container for the model-ready datasets produced by Phase 2."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: List[str] = field(default_factory=list)
    scaler: RobustScaler | None = None

    def as_tuple(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
        """Expose the classic 7-element tuple expected by the notebooks."""
        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
            self.feature_names,
        )


def _require_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Return the subset of ``columns`` actually present in ``df``."""
    return [c for c in columns if c in df.columns]


def create_features(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Derive the engineered features, skipping any that lack source columns.

    Each feature is guarded so a dataset that does not expose, say,
    ``horas_trabajo_año`` still produces a usable frame instead of raising.
    """
    df = df.copy()
    created = 0

    if {config.days_worked_col} <= set(df.columns):
        df["antiguedad_anios"] = df[config.days_worked_col] / 365.25
        df["antiguedad_log"] = np.log1p(df["antiguedad_anios"])
        created += 2

    if config.yearly_hours_col in df.columns:
        df["carga_laboral_diaria"] = df[config.yearly_hours_col] / 365.25
        created += 1

    if {"carga_laboral_diaria", config.salary_col} <= set(df.columns):
        df["ratio_carga_salarial"] = df["carga_laboral_diaria"] / (
            df[config.salary_col] + 1
        )
        created += 1

    health_inputs = [
        config.medical_visits_col,
        config.absenteeism_col,
        config.stress_col,
    ]
    if set(health_inputs) <= set(df.columns):
        df["indice_salud_ocupacional"] = (
            df[config.medical_visits_col] * 0.3
            + df[config.absenteeism_col] * 0.4
            + df[config.stress_col] * 0.3
        )
        created += 1

    if {config.climate_col, config.stress_col} <= set(df.columns):
        df["clima_estres_interaction"] = (
            df[config.climate_col] * df[config.stress_col]
        )
        created += 1

    if config.salary_col in df.columns:
        df["categoria_sueldo"] = pd.cut(
            df[config.salary_col],
            bins=list(config.salary_bins),
            labels=list(config.salary_labels),
        )
        created += 1

    if config.age_col in df.columns:
        df["rango_edad"] = pd.cut(
            df[config.age_col],
            bins=list(config.age_bins),
            labels=list(config.age_labels),
        )
        created += 1

    print(f"   ✅ {created} nuevas variables creadas")
    return df


def encode_categoricals(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """One-hot encode low-cardinality categoricals, target-encode the rest."""
    df_proc = df.copy()
    cat_vars = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_vars = [c for c in cat_vars if c != config.target_col]

    low_card = [
        c for c in cat_vars if df_proc[c].nunique() < config.one_hot_max_cardinality
    ]
    high_card = [
        c for c in cat_vars if df_proc[c].nunique() >= config.one_hot_max_cardinality
    ]

    df_enc = pd.get_dummies(df_proc, columns=low_card, drop_first=True)

    for col in high_card:
        target_means = df_proc.groupby(col)[config.target_col].mean()
        df_enc[f"{col}_target_enc"] = df_proc[col].map(target_means)
        df_enc = df_enc.drop(col, axis=1)

    print(
        f"    One-Hot: {len(low_card)} variables | "
        f"Target Encoding: {len(high_card)} variables"
    )
    return df_enc


def scale_features(
    X: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, RobustScaler]:
    """Scale numeric columns with RobustScaler and cast booleans to ints."""
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)

    print(f"    {len(numeric_cols)} variables escaladas")
    return X, scaler


def build_dataset(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, pd.Series, RobustScaler]:
    """Full feature matrix build: drop id, engineer, encode, scale."""
    df_proc = df.copy()
    if config.id_col in df_proc.columns:
        df_proc = df_proc.drop(columns=[config.id_col])

    df_proc = create_features(df_proc, config)
    df_enc = encode_categoricals(df_proc, config)

    X = df_enc.drop(config.target_col, axis=1)
    y = df_enc[config.target_col]

    X, scaler = scale_features(X, config)
    return X, y, scaler


def balance_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Apply SMOTE then a stratified 70/15/15 train/val/test split."""
    print("\n🔧 MANEJO DE DESBALANCE (SMOTE)")
    print(f"   Antes: Clase 0={(y == 0).sum()}, Clase 1={(y == 1).sum()}")
    smote = SMOTE(
        random_state=config.xgb_params.get("random_state", 42),
        sampling_strategy=config.smote_sampling_strategy,
    )
    X_res, y_res = smote.fit_resample(X, y)
    print(f"   Después: Clase 0={(y_res == 0).sum()}, Clase 1={(y_res == 1).sum()}")

    print("\n🔧 PARTICIÓN Train/Val/Test (70/15/15)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_res,
        y_res,
        test_size=config.test_size,
        random_state=config.xgb_params.get("random_state", 42),
        stratify=y_res,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.val_size,
        random_state=config.xgb_params.get("random_state", 42),
        stratify=y_temp,
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> PreparedData:
    """Phase 2 orchestrator: features -> encoding -> scaling -> SMOTE -> split."""
    print("\n" + "=" * 80)
    print("FASE 2: PREPROCESAMIENTO Y PREPARACIÓN DE DATOS")
    print("=" * 80)

    print("\n🔧 FEATURE ENGINEERING")
    X, y, scaler = build_dataset(df, config)

    X_train, X_val, X_test, y_train, y_val, y_test = balance_and_split(X, y, config)

    print("\n✅ FASE 2 COMPLETADA: Preprocesamiento")
    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=X.columns.tolist(),
        scaler=scaler,
    )


# Backwards-compatible alias matching the original function name.
fase_2_preprocesamiento = preprocess
