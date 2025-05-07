#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Imports from existing codebase
from src.analysis.data_processing import load_data, filter_df_for_modeling, add_cumulative_dose
from src.analysis.models.intact_model import IntactModel
from src.analysis.models.concentration_dependent_dna_model import ConcentrationDependentDNAModel
from src.analysis.evaluation import calculate_metrics, calculate_dna_metrics

# Import visualization helpers
from visualization_helpers import generate_plots


def run_standard_analysis(df_raw: pd.DataFrame, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run standard analysis with both intact and concentration-dependent DNA models

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "standard_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running standard analysis")
    print(f"Saving results to {output_dir}")

    # --- 1. Filter data for intact model fitting ---
    print("\n[1] Filtering data for intact model...")
    df_intact_fit = filter_df_for_modeling(df_raw)
    print(f"Filtered data to {len(df_intact_fit)} rows for intact model fitting")

    # --- 2. Fit intact model ---
    print("\n[2] Fitting intact model...")
    intact_model = IntactModel()
    intact_success = intact_model.fit(df_intact_fit)

    if not intact_success:
        print("Failed to fit intact model.")
        return {"error": "Failed to fit intact model"}

    k = intact_model.params["k"]
    alpha = intact_model.params["alpha"]
    print(f"Fitted intact model parameters: k={k:.3e}, alpha={alpha:.3f}")

    # --- 3. Generate intact predictions ---
    print("\n[3] Generating intact predictions...")
    df_with_intact_pred = intact_model.predict(df_raw)

    # --- 4. Calculate metrics for intact model ---
    intact_metrics = calculate_metrics(df_with_intact_pred)
    print(
        f"Intact model metrics: R²={intact_metrics.get('R²', 'N/A'):.4f}, RMSE={intact_metrics.get('RMSE', 'N/A'):.4f}")

    # --- 5. Fit DNA model ---
    print("\n[4] Fitting concentration-dependent DNA model...")
    dna_model = ConcentrationDependentDNAModel()
    dna_success = dna_model.fit(df_with_intact_pred)

    if not dna_success:
        print("Failed to fit DNA model.")
        return {
            "intact_model": {
                "parameters": intact_model.params,
                "metrics": intact_metrics
            },
            "error": "Failed to fit DNA model"
        }

    # Print DNA model parameters
    print("DNA model parameters:")
    for key, value in dna_model.params.items():
        print(f"  {key}: {value}")

    # --- 6. Generate DNA predictions ---
    print("\n[5] Generating DNA predictions...")
    df_with_dna_pred = dna_model.predict(df_with_intact_pred)

    # --- 7. Calculate metrics for DNA model ---
    dna_metrics = calculate_dna_metrics(df_with_dna_pred)
    print(f"DNA model metrics: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.4f}")

    # --- 8. Generate extensive plots ---
    print("\n[6] Generating extensive plots...")
    generate_plots(df_raw, df_with_intact_pred, df_with_dna_pred, intact_model, dna_model, output_dir)

    # --- 9. Save results ---
    print("\n[7] Saving results...")
    results = {
        "intact_model": {
            "parameters": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in intact_model.params.items()},
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in intact_metrics.items()}
        },
        "dna_model": {
            "model_type": "concentration_dependent",
            "parameters": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in dna_model.params.items()},
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in dna_metrics.items()}
        }
    }

    # Save predictions
    pred_path = output_dir / "predictions.csv"
    df_with_dna_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Save results as JSON
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_path}")

    return results


if __name__ == "__main__":
    print("Running standalone standard analysis")
    data_path = PROJECT_ROOT / "data" / "scFv" / "scfv_lysis.xlsx"
    df_raw = load_data(data_path)
    run_standard_analysis(df_raw)