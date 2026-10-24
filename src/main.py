"""End-to-end Smart Cargo pipeline orchestrator.

Runs the four stages in order:

    raw data -> clean -> features -> train -> evaluate

Usage
-----
    python -m src.main                       # uses data/raw/base_proyecto_.xlsx
    python -m src.main --input my_data.xlsx  # explicit input path
    python -m src.main --no-save             # skip writing cleaned data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_PROCESSED_PATH, DEFAULT_RAW_PATH
from .data_processing import clean_dataset, load_dataset, save_processed
from .evaluate import run_metrics, run_results
from .feature_engineering import preprocess
from .train_model import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Cargo attrition pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to the raw dataset (.xlsx or .csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Where to persist the cleaned dataset.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the cleaned dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG

    print("=" * 80)
    print("SMART-CARGO")
    print("Sistema Predictivo de Retención de Talento")
    print("=" * 80)

    print("\n📂 Cargando y limpiando datos...")
    df = load_dataset(args.input)
    df = clean_dataset(df, config)
    if not args.no_save:
        save_processed(df, args.output)
        print(f"✅ Dataset limpio guardado en: {args.output}")

    prepared = preprocess(df, config)

    results, selected_features, X_val_sfs, y_val_sfs = run_training(
        prepared.X_train,
        prepared.X_val,
        prepared.X_test,
        prepared.y_train,
        prepared.y_val,
        prepared.y_test,
        config,
    )

    run_results(results, selected_features, X_val_sfs, y_val_sfs)
    run_metrics(results, y_val_sfs, config)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
