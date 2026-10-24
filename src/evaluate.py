"""Evaluation, visualization and interpretation (Phases 4 and 5).

Produces the comparative ROC / precision-recall curves, the confusion matrix
and feature importance for the winning model, the SHAP explanation, and the
master metrics table with the improvement over the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .config import DEFAULT_CONFIG, FIGURES_DIR, PipelineConfig
from .train_model import ModelResult

try:  # SHAP is optional: the pipeline still runs without it.
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the local environment
    SHAP_AVAILABLE = False


MODEL_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def _as_dict(results: Mapping[str, ModelResult]) -> Dict[str, dict]:
    """Normalize results to the plain-dict layout used across this module."""
    normalized: Dict[str, dict] = {}
    for name, res in results.items():
        normalized[name] = res.as_dict() if isinstance(res, ModelResult) else res
    return normalized


def _save(fig, output_dir: Optional[Path], name: str) -> None:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_model_comparison(
    results: Mapping[str, ModelResult],
    selected_features,
    y_val: pd.Series,
    output_dir: Optional[Path] = None,
    best_model: Optional[str] = None,
) -> None:
    """Phase 4 plots: ROC, PR curves, confusion matrix and importances."""
    results = _as_dict(results)
    best_model = best_model or select_best_model(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "FASE 4: RESULTADOS Y COMPARACIÓN DE MODELOS", fontsize=16, fontweight="bold"
    )

    ax1 = axes[0, 0]
    for (name, res), color in zip(results.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_val, res["y_proba"])
        ax1.plot(
            fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2, color=color
        )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Curvas ROC Comparativas")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for (name, res), color in zip(results.items(), MODEL_COLORS):
        precision, recall, _ = precision_recall_curve(y_val, res["y_proba"])
        ax2.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={res['pr_auc']:.3f})",
            linewidth=2,
            color=color,
        )
    ax2.axhline(y=y_val.mean(), color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Curvas Precision-Recall")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    winner = results[best_model]
    ax3 = axes[1, 0]
    y_pred = (winner["y_proba"] >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_title(f"Matriz de Confusión - {best_model}")
    ax3.set_xlabel("Predicción")
    ax3.set_ylabel("Real")

    ax4 = axes[1, 1]
    importance = pd.DataFrame(
        {
            "feature": list(selected_features),
            "importance": winner["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=True)
    ax4.barh(importance["feature"], importance["importance"], color="steelblue")
    ax4.set_xlabel("Importancia")
    ax4.set_title(f"Feature Importance - {best_model}")

    plt.tight_layout()
    _save(fig, output_dir, "phase4_model_comparison")
    plt.show()


def plot_shap(
    winner: dict,
    X_val: pd.DataFrame,
    selected_features,
    output_dir: Optional[Path] = None,
    model_name: str = "mejor modelo",
) -> None:
    """Render the SHAP summary plot for the winning model when available."""
    if not SHAP_AVAILABLE:
        print("\n⚠️ SHAP no está instalado; se omite la interpretación.")
        return

    print("\nGenerando interpretación SHAP...")
    # TreeExplainer only understands tree ensembles; fall back to the
    # model-agnostic explainer for anything else (e.g. LogisticRegression).
    try:
        explainer = shap.TreeExplainer(winner["model"])
        shap_values = explainer.shap_values(X_val)
    except Exception:
        explainer = shap.Explainer(winner["model"], X_val)
        shap_values = explainer(X_val)

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val, feature_names=list(selected_features), show=False)
    plt.title(f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "phase4_shap_summary")
    plt.show()


def run_results(
    results: Mapping[str, ModelResult],
    selected_features,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Optional[Path] = FIGURES_DIR,
) -> None:
    """Phase 4 orchestrator: comparative plots plus SHAP interpretation."""
    print("\n" + "=" * 80)
    print("FASE 4: RESULTADOS Y VISUALIZACIONES")
    print("=" * 80)

    normalized = _as_dict(results)
    best_model = select_best_model(normalized)
    plot_model_comparison(
        normalized, selected_features, y_val, output_dir, best_model=best_model
    )
    plot_shap(
        normalized[best_model],
        X_val,
        selected_features,
        output_dir,
        model_name=best_model,
    )

    print("\n✅ FASE 4 COMPLETADA: Resultados")


def build_comparison_table(
    results: Mapping[str, ModelResult], y_val: pd.Series
) -> pd.DataFrame:
    """Build the master metrics table including the naive baseline row."""
    results = _as_dict(results)
    baseline_proba = np.ones(len(y_val)) * y_val.mean()
    baseline_pr = average_precision_score(y_val, baseline_proba)

    return pd.DataFrame(
        {
            "Modelo": ["Baseline"] + list(results.keys()),
            "PR-AUC": [baseline_pr] + [r["pr_auc"] for r in results.values()],
            "F1-Score": [0.0] + [r["f1"] for r in results.values()],
            "ROC-AUC": [0.5] + [r["roc_auc"] for r in results.values()],
        }
    )


def select_best_model(results: Mapping[str, ModelResult]) -> str:
    """Pick the model with the highest PR-AUC (primary imbalanced metric)."""
    results = _as_dict(results)
    return max(results, key=lambda name: results[name]["pr_auc"])


def run_metrics(
    results: Mapping[str, ModelResult],
    y_val: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Phase 5 orchestrator: master table, improvement vs baseline, winner."""
    print("\n" + "=" * 80)
    print("FASE 5: CUANTIFICACIÓN Y COMPARACIÓN DE MÉTRICAS")
    print("=" * 80)

    results = _as_dict(results)
    baseline_pr = average_precision_score(
        y_val, np.ones(len(y_val)) * y_val.mean()
    )

    comparison = build_comparison_table(results, y_val)
    print("\nTABLA COMPARATIVA MAESTRA:")
    print("=" * 80)
    print(comparison.to_string(index=False, float_format="%.4f"))

    print("\nMEJORA VS BASELINE:")
    for name, res in results.items():
        mejora = (res["pr_auc"] - baseline_pr) / baseline_pr * 100
        print(f"   {name}: +{mejora:.1f}% en PR-AUC")

    best = select_best_model(results)
    print("\n" + "=" * 80)
    print(f"🏆 MODELO GANADOR: {best}")
    print("=" * 80)
    print(
        f"""
    JUSTIFICACIÓN:
    • PR-AUC: {results[best]['pr_auc']:.4f}
    • F1-Score: {results[best]['f1']:.4f}
    • ROC-AUC: {results[best]['roc_auc']:.4f}
    • Alta interpretabilidad con SHAP values
    • Manejo nativo de desbalance
    """
    )

    print("\n✅ FASE 5 COMPLETADA: Cuantificación y Comparación")
    return comparison


# Backwards-compatible aliases matching the original function names.
fase_4_resultados = run_results
fase_5_metricas = run_metrics
