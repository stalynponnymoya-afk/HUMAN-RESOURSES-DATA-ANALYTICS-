"""Model training and selection (Phase 3).

Trains three candidates on a shared feature subset selected by a baseline
XGBoost model: XGBoost, Random Forest and Logistic Regression. Each model is
scored on the validation split with PR-AUC (the primary metric for imbalanced
attrition data), F1 and ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import DEFAULT_CONFIG, PipelineConfig


@dataclass
class ModelResult:
    """A trained model together with its validation metrics."""

    name: str
    model: object
    pr_auc: float
    f1: float
    roc_auc: float
    y_proba: "pd.Series"

    def as_dict(self) -> Dict[str, object]:
        """Return the dict layout used by the evaluation module."""
        return {
            "model": self.model,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "y_proba": self.y_proba,
        }


def _scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio of negative to positive samples, guarding against zero positives."""
    positives = (y_train == 1).sum()
    negatives = (y_train == 0).sum()
    if positives == 0:
        return 1.0
    return negatives / positives


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> List[str]:
    """Rank features with a baseline XGBoost model and keep the top K."""
    print(f"\n🔍 Seleccionando las top {config.top_k_features} características con XGBoost...")
    base_params = dict(config.xgb_base_params)
    base_params["scale_pos_weight"] = _scale_pos_weight(y_train)

    xgb_base = xgb.XGBClassifier(**base_params)
    xgb_base.fit(X_train, y_train)

    importance = (
        pd.DataFrame(
            {"feature": X_train.columns, "importance": xgb_base.feature_importances_}
        )
        .sort_values("importance", ascending=False)
    )
    selected = importance.head(config.top_k_features)["feature"].tolist()
    print(f"✅ {len(selected)} características seleccionadas")
    return selected


def _evaluate_on_val(
    name: str, model: object, X_val: pd.DataFrame, y_val: pd.Series
) -> ModelResult:
    """Score ``model`` on the validation split and print a short report."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    result = ModelResult(
        name=name,
        model=model,
        pr_auc=average_precision_score(y_val, y_proba),
        f1=f1_score(y_val, y_pred),
        roc_auc=roc_auc_score(y_val, y_proba),
        y_proba=y_proba,
    )
    print(f"PR-AUC: {result.pr_auc:.4f}")
    print(f"F1-Score: {result.f1:.4f}")
    print(f"ROC-AUC: {result.roc_auc:.4f}")
    return result


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ModelResult:
    """Model 1: gradient boosting with L1/L2 regularization."""
    print("\n" + "=" * 60)
    print("MODELO 1: XGBoost con SFS")
    print("=" * 60)
    print("Justificación: Gradient boosting con regularización L1/L2,")
    print("manejo de desbalance y alta interpretabilidad.")

    params = dict(config.xgb_params)
    params["scale_pos_weight"] = _scale_pos_weight(y_train)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return _evaluate_on_val("XGBoost+SFS", model, X_val, y_val)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ModelResult:
    """Model 2: bagged trees, robust to outliers and variance."""
    print("\n" + "=" * 60)
    print("MODELO 2: Random Forest")
    print("=" * 60)
    print("Justificación: Ensemble robusto, reduce varianza,")
    print("maneja bien outliers y proporciona feature importance.")

    model = RandomForestClassifier(**config.rf_params)
    model.fit(X_train, y_train)
    return _evaluate_on_val("Random Forest", model, X_val, y_val)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ModelResult:
    """Model 3: interpretable linear baseline."""
    print("\n" + "=" * 60)
    print("MODELO 3: Regresión Logística")
    print("=" * 60)
    print("Justificación: Modelo lineal interpretable, rápido,")
    print("sirve como baseline para comparación.")

    model = LogisticRegression(**config.lr_params)
    model.fit(X_train, y_train)
    return _evaluate_on_val("Regresión Logística", model, X_val, y_val)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selected_features: List[str],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Dict[str, ModelResult]:
    """Train all three candidates on ``selected_features``."""
    X_train_sfs = X_train[selected_features]
    X_val_sfs = X_val[selected_features]

    results = {
        "XGBoost+SFS": train_xgboost(
            X_train_sfs, y_train, X_val_sfs, y_val, config
        ),
        "Random Forest": train_random_forest(
            X_train_sfs, y_train, X_val_sfs, y_val, config
        ),
        "Regresión Logística": train_logistic_regression(
            X_train_sfs, y_train, X_val_sfs, y_val, config
        ),
    }
    return results


def run_training(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[Dict[str, ModelResult], List[str], pd.DataFrame, pd.Series]:
    """Phase 3 orchestrator: select features, train, return results."""
    print("\n" + "=" * 80)
    print("FASE 3: MODELAMIENTO")
    print("=" * 80)

    selected_features = select_features(X_train, y_train, config)
    results = train_models(
        X_train, y_train, X_val, y_val, selected_features, config
    )
    X_val_sfs = X_val[selected_features]

    print("\n✅ FASE 3 COMPLETADA: Modelamiento")
    return results, selected_features, X_val_sfs, y_val


# Backwards-compatible alias matching the original function name.
fase_3_modelamiento = run_training
