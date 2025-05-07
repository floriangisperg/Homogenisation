#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Imports from existing codebase
from src.analysis.data_processing import load_data, filter_df_for_modeling
from src.analysis.models.intact_model import IntactModel
from src.analysis.models.concentration_dependent_dna_model import (
    ConcentrationDependentDNAModel,
)
from src.analysis.evaluation import calculate_metrics, calculate_dna_metrics
from src.analysis.visualization import VisualizationManager

# Import visualization helpers
from visualization_helpers import (
    visualize_fold,
    plot_metrics_comparison,
    plot_parameter_distributions,
)


def run_loocv(
    df_raw: pd.DataFrame, output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run Leave-One-Out Cross-Validation (LOOCV) with both models.

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)

    Returns:
        Dictionary with LOOCV results
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "loocv_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Leave-One-Out Cross-Validation")
    print(f"Saving results to {output_dir}")

    # Handle NA values in experiment_id before sorting
    df_valid = df_raw.dropna(subset=["experiment_id"]).copy()
    exp_ids = sorted(
        [int(x) for x in df_valid["experiment_id"].unique() if pd.notna(x)]
    )
    n_folds = len(exp_ids)

    if n_folds < 2:
        print(f"Error: Need at least 2 experiments for LOOCV. Found {n_folds}.")
        return {"error": "Insufficient experiments for LOOCV"}

    print(f"Found {n_folds} experiments for LOOCV: {exp_ids}")

    # Store results for each fold
    fold_results = []
    all_test_predictions = []
    fold_parameters = {"intact": [], "dna": []}

    # Create tables for direct comparison of metrics across folds
    fold_intact_metrics_table = []
    fold_dna_metrics_table = []

    # Run LOOCV
    for i, test_exp_id in enumerate(exp_ids):
        fold_num = i + 1
        print(
            f"\n--- LOOCV Fold {fold_num}/{n_folds}: Test Experiment {test_exp_id} ---"
        )

        # Split data - use df_valid instead of df_raw to avoid NA issues
        train_df = df_valid[df_valid["experiment_id"] != test_exp_id].copy()
        test_df = df_valid[df_valid["experiment_id"] == test_exp_id].copy()

        # Check if we have enough data
        if train_df.empty or test_df.empty:
            print(f"Warning: Empty train or test set for fold {fold_num}. Skipping.")
            continue

        # Filter training data for intact model fitting
        train_intact_fit = filter_df_for_modeling(train_df)

        # 1. Fit intact model on training data
        print(f"Fitting intact model on training data...")
        intact_model = IntactModel()
        intact_success = intact_model.fit(train_intact_fit)

        if not intact_success:
            print(f"Failed to fit intact model for fold {fold_num}. Skipping.")
            continue

        # Save parameters
        fold_parameters["intact"].append(
            {
                "fold": fold_num,
                "test_exp_id": test_exp_id,
                "k": intact_model.params["k"],
                "alpha": intact_model.params["alpha"],
            }
        )

        # 2. Generate intact predictions for training data
        train_with_intact_pred = intact_model.predict(train_df)

        # 3. Fit DNA model on training data
        print(f"Fitting DNA model on training data...")
        dna_model = ConcentrationDependentDNAModel()
        dna_success = dna_model.fit(train_with_intact_pred)

        if not dna_success:
            print(f"Failed to fit DNA model for fold {fold_num}. Skipping.")
            continue

        # Save parameters
        fold_parameters["dna"].append(
            {
                "fold": fold_num,
                "test_exp_id": test_exp_id,
                **{
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in dna_model.params.items()
                },
            }
        )

        # 4. Generate predictions for test data
        print(f"Generating predictions for test data...")
        test_with_intact_pred = intact_model.predict(test_df)
        test_with_dna_pred = dna_model.predict(test_with_intact_pred)

        # Add fold and test info to predictions
        test_with_dna_pred["fold"] = fold_num
        test_with_dna_pred["is_test"] = True

        # 5. Calculate metrics
        intact_metrics = calculate_metrics(test_with_intact_pred)
        dna_metrics = calculate_dna_metrics(test_with_dna_pred)

        # Add to metrics tables
        fold_intact_metrics_table.append(
            {
                "fold": fold_num,
                "test_exp_id": test_exp_id,
                **{k: v for k, v in intact_metrics.items() if k != "N_valid"},
            }
        )

        fold_dna_metrics_table.append(
            {
                "fold": fold_num,
                "test_exp_id": test_exp_id,
                **{k: v for k, v in dna_metrics.items() if k != "N_valid"},
            }
        )

        # Display key metrics including new normalized metrics
        print(
            f"Test metrics - Intact: R²={intact_metrics.get('R²', 'N/A'):.4f}, RMSE={intact_metrics.get('RMSE', 'N/A'):.4f}"
        )
        if "NRMSE_mean" in dna_metrics and pd.notna(dna_metrics["NRMSE_mean"]):
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}, NRMSE(mean)={dna_metrics.get('NRMSE_mean', 'N/A'):.3f}"
            )
        else:
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}"
            )

        # 6. Generate fold-specific plots directly in the output directory
        print(f"Generating fold-specific plots...")
        visualize_fold(
            test_with_dna_pred, output_dir, fold_num=fold_num, test_exp_id=test_exp_id
        )

        # 7. Store results for this fold
        fold_result = {
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            "intact_metrics": intact_metrics,
            "dna_metrics": dna_metrics,
            "intact_params": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in intact_model.params.items()
            },
            "dna_params": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in dna_model.params.items()
            },
        }

        fold_results.append(fold_result)
        all_test_predictions.append(test_with_dna_pred)

    # Check if we have any valid folds
    if not fold_results:
        print("Error: No valid LOOCV folds completed.")
        return {"error": "No valid LOOCV folds"}

    # Combine all test predictions
    print("\nCombining all test predictions...")
    all_preds_df = pd.concat(all_test_predictions, ignore_index=True)

    # Calculate overall metrics
    overall_intact_metrics = calculate_metrics(all_preds_df)
    overall_dna_metrics = calculate_dna_metrics(all_preds_df)

    # Print overall metrics including normalized metrics
    print(f"Overall LOOCV metrics:")
    print(
        f"  Intact: R²={overall_intact_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_intact_metrics.get('RMSE', 'N/A'):.4f}"
    )
    print(
        f"  DNA: R²={overall_dna_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_dna_metrics.get('RMSE', 'N/A'):.1f}"
    )
    if "NRMSE_mean" in overall_dna_metrics and pd.notna(
        overall_dna_metrics["NRMSE_mean"]
    ):
        print(
            f"  DNA normalized: NRMSE(mean)={overall_dna_metrics.get('NRMSE_mean', 'N/A'):.3f}, NRMSE(range)={overall_dna_metrics.get('NRMSE_range', 'N/A'):.3f}"
        )

    # Calculate average parameters
    avg_params = {}
    for model_type in ["intact", "dna"]:
        if not fold_parameters[model_type]:
            continue

        avg_params[model_type] = {}
        for param in fold_parameters[model_type][0].keys():
            if param in ["fold", "test_exp_id"]:
                continue

            values = [
                fold[param] for fold in fold_parameters[model_type] if param in fold
            ]
            if values:
                avg_params[model_type][param] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

    # After combining all test predictions, add these visualization calls:
    print("\nGenerating overall LOOCV plots...")
    viz = VisualizationManager(output_dir)

    # Intact parity plot
    viz.plot_intact_parity(all_preds_df, title="LOOCV: Intact Fraction")

    # DNA parity plots (both log and linear scales)
    viz.plot_dna_parity(all_preds_df, title="LOOCV: DNA Concentration", log_scale=True)
    viz.plot_dna_parity(all_preds_df, title="LOOCV: DNA Concentration", log_scale=False)

    # Combined plot of intact fraction and DNA with both scale options
    viz.plot_combined_intact_dna_vs_process(
        all_preds_df,
        log_scale_dna=True,
        title="LOOCV: Intact Fraction and DNA",
        cv_mode=True,
    )
    viz.plot_combined_intact_dna_vs_process(
        all_preds_df,
        log_scale_dna=False,
        title="LOOCV: Intact Fraction and DNA",
        cv_mode=True,
    )

    # Create metrics comparison plots
    try:
        plot_metrics_comparison(
            fold_intact_metrics_table, fold_dna_metrics_table, output_dir
        )
    except Exception as e:
        print(f"Warning: Could not create metrics comparison plots: {e}")

    # Save metrics tables as CSVs for easy analysis
    # Save intact metrics table
    if fold_intact_metrics_table:
        intact_metrics_df = pd.DataFrame(fold_intact_metrics_table)
        intact_metrics_path = output_dir / "intact_metrics_by_fold.csv"
        intact_metrics_df.to_csv(intact_metrics_path, index=False)
        print(f"Saved intact metrics by fold to {intact_metrics_path}")

    # Save DNA metrics table
    if fold_dna_metrics_table:
        dna_metrics_df = pd.DataFrame(fold_dna_metrics_table)
        dna_metrics_path = output_dir / "dna_metrics_by_fold.csv"
        dna_metrics_df.to_csv(dna_metrics_path, index=False)
        print(f"Saved DNA metrics by fold to {dna_metrics_path}")

    # Visualize metrics across folds
    try:
        plot_metrics_comparison(
            fold_intact_metrics_table, fold_dna_metrics_table, output_dir
        )
    except Exception as e:
        print(f"Warning: Could not create metrics comparison plots: {e}")

    # Save all test predictions
    pred_path = output_dir / "loocv_predictions.csv"
    all_preds_df.to_csv(pred_path, index=False)
    print(f"Saved all LOOCV predictions to {pred_path}")

    # Save overall results
    loocv_results = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "overall_intact_metrics": {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in overall_intact_metrics.items()
        },
        "overall_dna_metrics": {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in overall_dna_metrics.items()
        },
        "average_parameters": avg_params,
    }

    results_path = output_dir / "loocv_results.json"
    with open(results_path, "w") as f:
        json.dump(loocv_results, f, indent=4)
    print(f"Saved LOOCV results to {results_path}")

    # Plot parameter distributions
    try:
        plot_parameter_distributions(fold_parameters, output_dir)
    except Exception as e:
        print(f"Warning: Could not plot parameter distributions: {e}")

    return loocv_results


if __name__ == "__main__":
    print("Running standalone LOOCV analysis")
    data_path = PROJECT_ROOT / "data" / "scFv" / "scfv_lysis.xlsx"
    df_raw = load_data(data_path)
    run_loocv(df_raw)
