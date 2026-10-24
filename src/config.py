"""Central configuration for the Smart Cargo attrition pipeline.

Every dataset-specific name (column names, grouping keys, thresholds) lives
here instead of being hard-coded across the code base. That way a change in
the source spreadsheet only requires editing this file, not the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_RAW_PATH = RAW_DATA_DIR / "base_proyecto_.xlsx"
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_DIR / "dataset_clean.csv"

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42


@dataclass
class PipelineConfig:
    """Runtime parameters for the end-to-end pipeline.

    Attributes are grouped by pipeline stage so each module only needs to read
    the slice it cares about. All names default to the Smart Cargo dataset but
    can be overridden for a different schema without touching the logic.
    """

    # -- schema -----------------------------------------------------------
    id_col: str = "id"
    target_col: str = "DESERCION"
    salary_col: str = "sueldo"
    cargo_col: str = "cargo"
    group_col: str = "empresa_gestion"
    area_col: str = "area"
    age_col: str = "edad"

    # -- data_processing --------------------------------------------------
    # Pattern used to detect "director" roles without hard-coding the exact
    # label, so a renamed title such as "Managing Director" still matches.
    director_pattern: str = "director"
    # Minimum number of observed salaries a cargo needs before we trust its
    # mode for imputation. Guards against a single observation defining the
    # whole group.
    min_salary_observations: int = 1
    # Numeric columns inspected for outliers (IQR method) during the EDA.
    outlier_cols: Sequence[str] = ("sueldo", "dias_trabajo", "edad")

    # -- feature_engineering ---------------------------------------------
    # Columns required to build the engineered features. If any is missing the
    # corresponding feature is skipped instead of raising.
    days_worked_col: str = "dias_trabajo"
    yearly_hours_col: str = "horas_trabajo_año"
    medical_visits_col: str = "total_atenciones_medicas"
    absenteeism_col: str = "dias_ausentismo"
    stress_col: str = "estres(1-3)"
    climate_col: str = "clima(1-10)"

    salary_bins: Sequence[float] = (0, 500, 1000, 2000, float("inf"))
    salary_labels: Sequence[str] = ("Bajo", "Medio-Bajo", "Medio-Alto", "Alto")
    age_bins: Sequence[float] = (0, 25, 35, 45, float("inf"))
    age_labels: Sequence[str] = ("Joven", "Adulto_Joven", "Adulto", "Maduro")
    # Cardinality below which a categorical column gets one-hot encoding;
    # anything at or above is target-encoded.
    one_hot_max_cardinality: int = 10

    # -- modeling ---------------------------------------------------------
    top_k_features: int = 15
    smote_sampling_strategy: float = 0.5
    test_size: float = 0.30
    val_size: float = 0.50

    xgb_base_params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
    })
    xgb_params: Dict[str, object] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
    })
    rf_params: Dict[str, object] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })
    lr_params: Dict[str, object] = field(default_factory=lambda: {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    })

    def numeric_feature_columns(self) -> List[str]:
        """Columns consumed by the engineered numeric features."""
        return [
            self.days_worked_col,
            self.yearly_hours_col,
            self.medical_visits_col,
            self.absenteeism_col,
            self.stress_col,
            self.climate_col,
            self.salary_col,
            self.age_col,
        ]


DEFAULT_CONFIG = PipelineConfig()
