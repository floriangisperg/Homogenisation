#!/usr/bin/env python3
# scripts/run_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import argparse
import time

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scFv"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# --- Imports ---
from analysis.data_processing import load_data, filter_df_for_modeling, add_cumulative_dose
from analysis.models.intact_model import IntactModel
from analysis.models.concentration_dependent_dna_model import ConcentrationDependentDNAModel
from analysis.evaluation import calculate_metrics, calculate_dna_metrics
from analysis.visualization import VisualizationManager

# --- Constants ---
DATA_FILENAME = "scfv_lysis.xlsx"


# --- Core Analysis Functions ---
def run_standard_analysis(df_raw, output_dir=None):
    """
    Run standard analysis with both intact and concentration-dependent DNA models

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "standard_analysis"

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


# Fix for the experiment_id NA issue in run_loocv function:

def run_loocv(df_raw, output_dir=None):
    """
    Run Leave-One-Out Cross-Validation (LOOCV) with both models,
    including normalized metrics for better interpretation

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)

    Returns:
        Dictionary with enhanced LOOCV results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "loocv_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Leave-One-Out Cross-Validation")
    print(f"Saving results to {output_dir}")

    # Handle NA values in experiment_id before sorting
    df_valid = df_raw.dropna(subset=['experiment_id']).copy()
    exp_ids = sorted([int(x) for x in df_valid['experiment_id'].unique() if pd.notna(x)])
    n_folds = len(exp_ids)

    if n_folds < 2:
        print(f"Error: Need at least 2 experiments for LOOCV. Found {n_folds}.")
        return {"error": "Insufficient experiments for LOOCV"}

    print(f"Found {n_folds} experiments for LOOCV: {exp_ids}")

    # Create subdirectories
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    # Store results for each fold
    fold_results = []
    all_test_predictions = []
    fold_parameters = {"intact": [], "dna": []}

    # Create tables for direct comparison of metrics across folds
    fold_intact_metrics_table = []
    fold_dna_metrics_table = []

    # Run LOOCV
    for i, test_exp_id in enumerate(exp_ids):
        print(f"\n--- LOOCV Fold {i + 1}/{n_folds}: Test Experiment {test_exp_id} ---")

        # Create fold-specific directory
        fold_dir = folds_dir / f"fold_{i + 1}_exp_{test_exp_id}"
        fold_dir.mkdir(exist_ok=True)

        # Split data - use df_valid instead of df_raw to avoid NA issues
        train_df = df_valid[df_valid['experiment_id'] != test_exp_id].copy()
        test_df = df_valid[df_valid['experiment_id'] == test_exp_id].copy()

        # Check if we have enough data
        if train_df.empty or test_df.empty:
            print(f"Warning: Empty train or test set for fold {i + 1}. Skipping.")
            continue

        # Filter training data for intact model fitting
        train_intact_fit = filter_df_for_modeling(train_df)

        # 1. Fit intact model on training data
        print(f"Fitting intact model on training data...")
        intact_model = IntactModel()
        intact_success = intact_model.fit(train_intact_fit)

        if not intact_success:
            print(f"Failed to fit intact model for fold {i + 1}. Skipping.")
            continue

        # Save parameters
        fold_parameters["intact"].append({
            "fold": i + 1,
            "test_exp_id": test_exp_id,
            "k": intact_model.params["k"],
            "alpha": intact_model.params["alpha"]
        })

        # 2. Generate intact predictions for training data
        train_with_intact_pred = intact_model.predict(train_df)

        # 3. Fit DNA model on training data
        print(f"Fitting DNA model on training data...")
        dna_model = ConcentrationDependentDNAModel()
        dna_success = dna_model.fit(train_with_intact_pred)

        if not dna_success:
            print(f"Failed to fit DNA model for fold {i + 1}. Skipping.")
            continue

        # Save parameters
        fold_parameters["dna"].append({
            "fold": i + 1,
            "test_exp_id": test_exp_id,
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v
               for k, v in dna_model.params.items()}
        })

        # 4. Generate predictions for test data
        print(f"Generating predictions for test data...")
        test_with_intact_pred = intact_model.predict(test_df)
        test_with_dna_pred = dna_model.predict(test_with_intact_pred)

        # Add fold and test info to predictions
        test_with_dna_pred['fold'] = i + 1
        test_with_dna_pred['is_test'] = True

        # 5. Calculate metrics
        intact_metrics = calculate_metrics(test_with_intact_pred)
        dna_metrics = calculate_dna_metrics(test_with_dna_pred)

        # Add to metrics tables
        fold_intact_metrics_table.append({
            "fold": i + 1,
            "test_exp_id": test_exp_id,
            **{k: v for k, v in intact_metrics.items() if k != "N_valid"}
        })

        fold_dna_metrics_table.append({
            "fold": i + 1,
            "test_exp_id": test_exp_id,
            **{k: v for k, v in dna_metrics.items() if k != "N_valid"}
        })

        # Display key metrics including new normalized metrics
        print(
            f"Test metrics - Intact: R²={intact_metrics.get('R²', 'N/A'):.4f}, RMSE={intact_metrics.get('RMSE', 'N/A'):.4f}")
        if 'NRMSE_mean' in dna_metrics and pd.notna(dna_metrics['NRMSE_mean']):
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}, NRMSE(mean)={dna_metrics.get('NRMSE_mean', 'N/A'):.3f}")
        else:
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}")

        # 6. Generate fold-specific plots
        print(f"Generating fold-specific plots...")
        visualize_fold(test_with_dna_pred, fold_dir, fold_num=i + 1, test_exp_id=test_exp_id)

        # 7. Store results for this fold
        fold_result = {
            "fold": i + 1,
            "test_exp_id": test_exp_id,
            "intact_metrics": intact_metrics,
            "dna_metrics": dna_metrics,
            "intact_params": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                              for k, v in intact_model.params.items()},
            "dna_params": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in dna_model.params.items()}
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
        f"  Intact: R²={overall_intact_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_intact_metrics.get('RMSE', 'N/A'):.4f}")
    print(f"  DNA: R²={overall_dna_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_dna_metrics.get('RMSE', 'N/A'):.1f}")
    if 'NRMSE_mean' in overall_dna_metrics and pd.notna(overall_dna_metrics['NRMSE_mean']):
        print(
            f"  DNA normalized: NRMSE(mean)={overall_dna_metrics.get('NRMSE_mean', 'N/A'):.3f}, NRMSE(range)={overall_dna_metrics.get('NRMSE_range', 'N/A'):.3f}")

    # Calculate average parameters
    avg_params = {}
    for model_type in ["intact", "dna"]:
        if not fold_parameters[model_type]:
            continue

        avg_params[model_type] = {}
        for param in fold_parameters[model_type][0].keys():
            if param in ["fold", "test_exp_id"]:
                continue

            values = [fold[param] for fold in fold_parameters[model_type] if param in fold]
            if values:
                avg_params[model_type][param] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }

    # Generate overall plots
    print("\nGenerating overall LOOCV plots...")
    viz = VisualizationManager(output_dir)

    # Intact parity plot
    viz.plot_intact_parity(all_preds_df, title="LOOCV Intact Fraction: Observed vs Predicted")

    # DNA parity plot
    viz.plot_dna_parity(all_preds_df, title="LOOCV DNA Concentration: Observed vs Predicted")

    # Save metrics tables as CSVs for easy analysis
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Save intact metrics table
    if fold_intact_metrics_table:
        intact_metrics_df = pd.DataFrame(fold_intact_metrics_table)
        intact_metrics_path = metrics_dir / "intact_metrics_by_fold.csv"
        intact_metrics_df.to_csv(intact_metrics_path, index=False)
        print(f"Saved intact metrics by fold to {intact_metrics_path}")

    # Save DNA metrics table
    if fold_dna_metrics_table:
        dna_metrics_df = pd.DataFrame(fold_dna_metrics_table)
        dna_metrics_path = metrics_dir / "dna_metrics_by_fold.csv"
        dna_metrics_df.to_csv(dna_metrics_path, index=False)
        print(f"Saved DNA metrics by fold to {dna_metrics_path}")

    # Visualize metrics across folds
    try:
        plot_metrics_comparison(fold_intact_metrics_table, fold_dna_metrics_table, metrics_dir)
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
        "overall_intact_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                   for k, v in overall_intact_metrics.items()},
        "overall_dna_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in overall_dna_metrics.items()},
        "average_parameters": avg_params
    }

    results_path = output_dir / "loocv_results.json"
    with open(results_path, 'w') as f:
        json.dump(loocv_results, f, indent=4)
    print(f"Saved LOOCV results to {results_path}")

    # Plot parameter distributions
    try:
        plot_parameter_distributions(fold_parameters, output_dir)
    except Exception as e:
        print(f"Warning: Could not plot parameter distributions: {e}")

    return loocv_results


def plot_metrics_comparison(intact_metrics_table, dna_metrics_table, output_dir):
    """
    Create visualization of metrics across different folds for comparison

    Args:
        intact_metrics_table: List of dictionaries with intact metrics by fold
        dna_metrics_table: List of dictionaries with DNA metrics by fold
        output_dir: Output directory
    """
    # Skip if we don't have enough data
    if not intact_metrics_table or not dna_metrics_table:
        return

    # 1. Plot intact model metrics comparison
    intact_metrics_to_plot = ['R²', 'RMSE', 'MAE']
    if intact_metrics_table:
        df_intact = pd.DataFrame(intact_metrics_table)
        df_intact = df_intact.set_index('test_exp_id')

        fig, axes = plt.subplots(1, len(intact_metrics_to_plot), figsize=(15, 5))

        for i, metric in enumerate(intact_metrics_to_plot):
            if metric in df_intact.columns:
                ax = axes[i]
                df_intact[metric].plot(kind='bar', ax=ax, color='royalblue')
                ax.set_title(f'Intact Model: {metric} by Experiment')
                ax.set_xlabel('Test Experiment ID')
                ax.set_ylabel(metric)
                ax.axhline(y=df_intact[metric].mean(), color='r', linestyle='--',
                           label=f'Mean: {df_intact[metric].mean():.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "intact_metrics_comparison.png", dpi=300)
        plt.close()

    # 2. Plot DNA model metrics comparison
    dna_metrics_to_plot = ['R²', 'RMSE', 'NRMSE_mean', 'MAPE']
    if dna_metrics_table:
        df_dna = pd.DataFrame(dna_metrics_table)
        df_dna = df_dna.set_index('test_exp_id')

        # Filter to experiments with valid metrics
        valid_metrics = df_dna['R²'].notna()
        df_dna_valid = df_dna[valid_metrics]

        if not df_dna_valid.empty:
            fig, axes = plt.subplots(1, len(dna_metrics_to_plot), figsize=(15, 5))

            for i, metric in enumerate(dna_metrics_to_plot):
                if metric in df_dna_valid.columns:
                    ax = axes[i]
                    df_dna_valid[metric].plot(kind='bar', ax=ax, color='firebrick')
                    ax.set_title(f'DNA Model: {metric} by Experiment')
                    ax.set_xlabel('Test Experiment ID')
                    ax.set_ylabel(metric)
                    ax.axhline(y=df_dna_valid[metric].mean(), color='b', linestyle='--',
                               label=f'Mean: {df_dna_valid[metric].mean():.4f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "dna_metrics_comparison.png", dpi=300)
            plt.close()


def visualize_fold(test_predictions, output_dir, fold_num, test_exp_id):
    """
    Generate fold-specific visualizations for LOOCV

    Args:
        test_predictions: DataFrame with predictions for test data
        output_dir: Output directory
        fold_num: Fold number
        test_exp_id: Test experiment ID
    """
    # Create visualization manager
    viz = VisualizationManager(output_dir)

    # Skip if we don't have enough data
    if test_predictions.empty:
        print(f"  Warning: No test predictions for fold {fold_num}.")
        return

    # 1. Plot process step vs. predictions (key plot)
    df_plot = test_predictions.copy()

    # Sort by process step order if needed
    step_order = {
        'resuspended biomass': 0,
        'initial lysis': 1,
        '1st wash': 2,
        '2nd wash': 3,
        '3rd wash': 4,
        '4th wash': 5
    }

    if 'process_step' in df_plot.columns:
        df_plot['step_order'] = df_plot['process_step'].map(
            lambda x: step_order.get(str(x).lower(), 999))
        df_plot = df_plot.sort_values('step_order')

    # Create plot showing observed vs. predicted over process steps (with log scale)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Intact fraction plot
    ax1 = axes[0]
    if 'observed_frac' in df_plot.columns and 'intact_frac_pred' in df_plot.columns:
        valid_obs = df_plot['observed_frac'].notna()
        valid_pred = df_plot['intact_frac_pred'].notna()

        if valid_obs.any():
            ax1.plot(df_plot.loc[valid_obs, 'process_step'],
                     df_plot.loc[valid_obs, 'observed_frac'],
                     'o-', color='blue', linewidth=2, markersize=8, label='Observed')

        if valid_pred.any():
            ax1.plot(df_plot.loc[valid_pred, 'process_step'],
                     df_plot.loc[valid_pred, 'intact_frac_pred'],
                     'x--', color='red', linewidth=2, markersize=8, label='Predicted')

        ax1.set_ylabel('Intact Fraction', fontsize=12)
        ax1.set_title(f'LOOCV Fold {fold_num}: Experiment {test_exp_id} Intact Fraction', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Set y-axis limits with small padding
        ax1.set_ylim(-0.05, 1.05)

    # DNA concentration plot with log scale
    ax2 = axes[1]
    if 'dna_conc' in df_plot.columns and 'dna_pred' in df_plot.columns:
        valid_obs = df_plot['dna_conc'] > 0
        valid_pred = df_plot['dna_pred'] > 0

        if valid_obs.any():
            ax2.plot(df_plot.loc[valid_obs, 'process_step'],
                     df_plot.loc[valid_obs, 'dna_conc'],
                     'o-', color='blue', linewidth=2, markersize=8, label='Observed')

        if valid_pred.any():
            ax2.plot(df_plot.loc[valid_pred, 'process_step'],
                     df_plot.loc[valid_pred, 'dna_pred'],
                     'x--', color='red', linewidth=2, markersize=8, label='Predicted')

        ax2.set_ylabel('DNA Concentration [ng/µL] (log scale)', fontsize=12)
        ax2.set_title(f'LOOCV Fold {fold_num}: Experiment {test_exp_id} DNA Concentration', fontsize=14)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Process Step', fontsize=12)

    plt.tight_layout()

    # Save figure - using clear naming in a single folder
    fig_path = output_dir / f"fold_{fold_num}_exp_{test_exp_id}_process_steps_log.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create second plot for DNA without log scale
    if 'dna_conc' in df_plot.columns and 'dna_pred' in df_plot.columns:
        # Only create this plot if we have valid DNA data
        valid_obs = df_plot['dna_conc'] > 0
        valid_pred = df_plot['dna_pred'] > 0

        if valid_obs.any() or valid_pred.any():
            fig, ax = plt.subplots(figsize=(10, 6))

            if valid_obs.any():
                ax.plot(df_plot.loc[valid_obs, 'process_step'],
                        df_plot.loc[valid_obs, 'dna_conc'],
                        'o-', color='blue', linewidth=2, markersize=8, label='Observed')

            if valid_pred.any():
                ax.plot(df_plot.loc[valid_pred, 'process_step'],
                        df_plot.loc[valid_pred, 'dna_pred'],
                        'x--', color='red', linewidth=2, markersize=8, label='Predicted')

            ax.set_ylabel('DNA Concentration [ng/µL] (linear scale)', fontsize=12)
            ax.set_title(f'LOOCV Fold {fold_num}: Experiment {test_exp_id} DNA Concentration', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Process Step', fontsize=12)

            plt.tight_layout()

            # Save figure with linear scale
            fig_path = output_dir / f"fold_{fold_num}_exp_{test_exp_id}_process_steps_linear.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    # 2. Create parity plots directly in the main folder

    # Intact parity plot
    if 'observed_frac' in df_plot.columns and 'intact_frac_pred' in df_plot.columns:
        try:
            viz.plot_intact_parity(df_plot,
                                   title=f"Fold {fold_num}: Exp {test_exp_id} Intact Fraction")

            # Rename the file to follow our naming convention
            default_path = output_dir / "intact_parity_plot.png"
            if default_path.exists():
                new_path = output_dir / f"fold_{fold_num}_exp_{test_exp_id}_intact_parity.png"
                default_path.rename(new_path)
        except Exception as e:
            print(f"  Warning: Could not create intact parity plot: {e}")

    # DNA parity plot
    if 'dna_conc' in df_plot.columns and 'dna_pred' in df_plot.columns:
        try:
            viz.plot_dna_parity(df_plot,
                                title=f"Fold {fold_num}: Exp {test_exp_id} DNA Concentration")

            # Rename the file to follow our naming convention
            default_path = output_dir / "dna_parity_plot_loglog.png"
            if default_path.exists():
                new_path = output_dir / f"fold_{fold_num}_exp_{test_exp_id}_dna_parity.png"
                default_path.rename(new_path)
        except Exception as e:
            print(f"  Warning: Could not create DNA parity plot: {e}")

    # 3. Save test predictions for this fold
    pred_path = output_dir / f"fold_{fold_num}_exp_{test_exp_id}_predictions.csv"
    test_predictions.to_csv(pred_path, index=False)


def run_loocv(df_raw, output_dir=None):
    """
    Run Leave-One-Out Cross-Validation (LOOCV) with both models,
    including normalized metrics for better interpretation.
    All results go in a single folder with clear naming conventions.

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)

    Returns:
        Dictionary with enhanced LOOCV results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "loocv_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Leave-One-Out Cross-Validation")
    print(f"Saving results to {output_dir}")

    # Handle NA values in experiment_id before sorting
    df_valid = df_raw.dropna(subset=['experiment_id']).copy()
    exp_ids = sorted([int(x) for x in df_valid['experiment_id'].unique() if pd.notna(x)])
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
        print(f"\n--- LOOCV Fold {fold_num}/{n_folds}: Test Experiment {test_exp_id} ---")

        # Split data - use df_valid instead of df_raw to avoid NA issues
        train_df = df_valid[df_valid['experiment_id'] != test_exp_id].copy()
        test_df = df_valid[df_valid['experiment_id'] == test_exp_id].copy()

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
        fold_parameters["intact"].append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            "k": intact_model.params["k"],
            "alpha": intact_model.params["alpha"]
        })

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
        fold_parameters["dna"].append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v
               for k, v in dna_model.params.items()}
        })

        # 4. Generate predictions for test data
        print(f"Generating predictions for test data...")
        test_with_intact_pred = intact_model.predict(test_df)
        test_with_dna_pred = dna_model.predict(test_with_intact_pred)

        # Add fold and test info to predictions
        test_with_dna_pred['fold'] = fold_num
        test_with_dna_pred['is_test'] = True

        # 5. Calculate metrics
        intact_metrics = calculate_metrics(test_with_intact_pred)
        dna_metrics = calculate_dna_metrics(test_with_dna_pred)

        # Add to metrics tables
        fold_intact_metrics_table.append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            **{k: v for k, v in intact_metrics.items() if k != "N_valid"}
        })

        fold_dna_metrics_table.append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            **{k: v for k, v in dna_metrics.items() if k != "N_valid"}
        })

        # Display key metrics including new normalized metrics
        print(
            f"Test metrics - Intact: R²={intact_metrics.get('R²', 'N/A'):.4f}, RMSE={intact_metrics.get('RMSE', 'N/A'):.4f}")
        if 'NRMSE_mean' in dna_metrics and pd.notna(dna_metrics['NRMSE_mean']):
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}, NRMSE(mean)={dna_metrics.get('NRMSE_mean', 'N/A'):.3f}")
        else:
            print(
                f"Test metrics - DNA: R²={dna_metrics.get('R²', 'N/A'):.4f}, RMSE={dna_metrics.get('RMSE', 'N/A'):.1f}")

        # 6. Generate fold-specific plots directly in the output directory
        print(f"Generating fold-specific plots...")
        visualize_fold(test_with_dna_pred, output_dir, fold_num=fold_num, test_exp_id=test_exp_id)

        # 7. Store results for this fold
        fold_result = {
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            "intact_metrics": intact_metrics,
            "dna_metrics": dna_metrics,
            "intact_params": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                              for k, v in intact_model.params.items()},
            "dna_params": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in dna_model.params.items()}
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
        f"  Intact: R²={overall_intact_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_intact_metrics.get('RMSE', 'N/A'):.4f}")
    print(f"  DNA: R²={overall_dna_metrics.get('R²', 'N/A'):.4f}, RMSE={overall_dna_metrics.get('RMSE', 'N/A'):.1f}")
    if 'NRMSE_mean' in overall_dna_metrics and pd.notna(overall_dna_metrics['NRMSE_mean']):
        print(
            f"  DNA normalized: NRMSE(mean)={overall_dna_metrics.get('NRMSE_mean', 'N/A'):.3f}, NRMSE(range)={overall_dna_metrics.get('NRMSE_range', 'N/A'):.3f}")

    # Calculate average parameters
    avg_params = {}
    for model_type in ["intact", "dna"]:
        if not fold_parameters[model_type]:
            continue

        avg_params[model_type] = {}
        for param in fold_parameters[model_type][0].keys():
            if param in ["fold", "test_exp_id"]:
                continue

            values = [fold[param] for fold in fold_parameters[model_type] if param in fold]
            if values:
                avg_params[model_type][param] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }

    # Generate overall plots
    print("\nGenerating overall LOOCV plots...")
    viz = VisualizationManager(output_dir)

    # Intact parity plot for all experiments
    viz.plot_intact_parity(all_preds_df, title="LOOCV Intact Fraction: Observed vs Predicted")

    # DNA parity plot for all experiments
    viz.plot_dna_parity(all_preds_df, title="LOOCV DNA Concentration: Observed vs Predicted")

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
        plot_metrics_comparison(fold_intact_metrics_table, fold_dna_metrics_table, output_dir)
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
        "overall_intact_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                   for k, v in overall_intact_metrics.items()},
        "overall_dna_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in overall_dna_metrics.items()},
        "average_parameters": avg_params
    }

    results_path = output_dir / "loocv_results.json"
    with open(results_path, 'w') as f:
        json.dump(loocv_results, f, indent=4)
    print(f"Saved LOOCV results to {results_path}")

    # Plot parameter distributions
    try:
        plot_parameter_distributions(fold_parameters, output_dir)
    except Exception as e:
        print(f"Warning: Could not plot parameter distributions: {e}")

    return loocv_results


def plot_parameter_distributions(fold_parameters, output_dir):
    """
    Plot distributions of parameters across LOOCV folds

    Args:
        fold_parameters: Dictionary with parameters for each fold
        output_dir: Output directory
    """
    for model_type in ["intact", "dna"]:
        if not fold_parameters[model_type]:
            continue

        # Create figure
        params = [p for p in fold_parameters[model_type][0].keys()
                  if p not in ["fold", "test_exp_id"]]

        n_params = len(params)
        if n_params == 0:
            continue

        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each parameter
        for i, param in enumerate(params):
            if i >= len(axes):
                break

            values = [fold[param] for fold in fold_parameters[model_type] if param in fold]
            if not values:
                continue

            ax = axes[i]

            # Create histogram
            ax.hist(values, bins=min(10, len(values)), alpha=0.7, color='steelblue')

            # Add mean line
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--',
                       label=f'Mean: {mean_val:.3g}')

            # Add standard deviation
            std_val = np.std(values)
            ax.fill_between([mean_val - std_val, mean_val + std_val], 0, ax.get_ylim()[1],
                            color='red', alpha=0.1,
                            label=f'Std: {std_val:.3g}')

            # Labels
            ax.set_xlabel(param)
            ax.set_ylabel('Count')
            ax.set_title(f'{param} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        # Save figure
        fig_title = f"{model_type.title()} Model Parameters Distribution"
        plt.suptitle(fig_title, fontsize=16, y=1.02)
        plt.tight_layout()

        fig_path = output_dir / f"{model_type}_parameter_distributions.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def generate_plots(df_raw, df_intact_pred, df_dna_pred, intact_model, dna_model, output_dir):
    """
    Generate extensive plots for both models

    Args:
        df_raw: Raw data DataFrame
        df_intact_pred: DataFrame with intact predictions
        df_dna_pred: DataFrame with DNA predictions
        intact_model: Fitted IntactModel
        dna_model: Fitted DNA model
        output_dir: Output directory for plots
    """
    # Create visualization manager
    viz = VisualizationManager(output_dir)

    # --- Intact Model Plots ---
    print("  Creating intact model plots...")

    # 1. Intact parity plot
    viz.plot_intact_parity(df_intact_pred, title="Intact Fraction: Observed vs Predicted")

    # 2. Overview fitted vs dose plot
    viz.plot_overview_fitted(df_intact_pred, intact_model.params["k"], intact_model.params["alpha"])

    # 3. Yield contour plots
    viz.plot_yield_contour(intact_model.params["k"], intact_model.params["alpha"], F0=1.0,
                           subdir="contour_plots",)

    # If frozen biomass exists, create separate contour plot
    if "frozen biomass" in df_raw["biomass_type"].str.lower().unique():
        frozen_data = df_raw[df_raw["biomass_type"].str.lower() == "frozen biomass"].copy()
        if not frozen_data.empty and "intact_biomass_percent" in frozen_data.columns:
            frozen_F0 = frozen_data["intact_biomass_percent"].iloc[0] / 100.0
            if 0 < frozen_F0 <= 1:
                viz.plot_yield_contour(intact_model.params["k"], intact_model.params["alpha"], F0=frozen_F0,
                                       subdir="contour_plots", )

    # 4. Residual plots
    viz.plot_residuals(df_intact_pred, 'observed_frac', 'intact_frac_pred',
                       title="Intact Fraction Residual Analysis")

    # --- DNA Model Plots ---
    print("  Creating DNA model plots...")

    # 1. DNA parity plot
    viz.plot_dna_parity(df_dna_pred, title="DNA Concentration: Observed vs Predicted")

    # 2. DNA residual plots (log scale)
    viz.plot_residuals(df_dna_pred, 'dna_conc', 'dna_pred',
                       title="DNA Concentration Residual Analysis", log_scale=True)

    # 3. Experiment-specific plots
    exp_ids = [x for x in df_dna_pred["experiment_id"].unique() if pd.notna(x)]
    for exp_id in exp_ids:
        try:
            # Process step vs. predictions
            exp_data = df_dna_pred[df_dna_pred["experiment_id"] == exp_id].copy()
            if not exp_data.empty and 'process_step' in exp_data.columns:
                # Sort by process step order if needed
                step_order = {
                    'resuspended biomass': 0,
                    'initial lysis': 1,
                    '1st wash': 2,
                    '2nd wash': 3,
                    '3rd wash': 4,
                    '4th wash': 5
                }
                exp_data['step_order'] = exp_data['process_step'].map(
                    lambda x: step_order.get(str(x).lower(), 999))
                exp_data = exp_data.sort_values('step_order')

                # Create figure
                fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

                # Intact fraction plot
                ax1 = axes[0]
                valid_obs = exp_data['observed_frac'].notna()
                valid_pred = exp_data['intact_frac_pred'].notna()

                if valid_obs.any():
                    ax1.plot(exp_data.loc[valid_obs, 'process_step'],
                             exp_data.loc[valid_obs, 'observed_frac'],
                             'o-', color='blue', linewidth=2, markersize=8, label='Observed')

                if valid_pred.any():
                    ax1.plot(exp_data.loc[valid_pred, 'process_step'],
                             exp_data.loc[valid_pred, 'intact_frac_pred'],
                             'x--', color='red', linewidth=2, markersize=8, label='Predicted')

                ax1.set_ylabel('Intact Fraction', fontsize=12)
                ax1.set_title(f'Experiment {exp_id}: Intact Fraction', fontsize=14)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(-0.05, 1.05)

                # DNA concentration plot
                ax2 = axes[1]
                valid_obs = exp_data['dna_conc'] > 0
                valid_pred = exp_data['dna_pred'] > 0

                if valid_obs.any():
                    ax2.plot(exp_data.loc[valid_obs, 'process_step'],
                             exp_data.loc[valid_obs, 'dna_conc'],
                             'o-', color='blue', linewidth=2, markersize=8, label='Observed')

                if valid_pred.any():
                    ax2.plot(exp_data.loc[valid_pred, 'process_step'],
                             exp_data.loc[valid_pred, 'dna_pred'],
                             'x--', color='red', linewidth=2, markersize=8, label='Predicted')

                ax2.set_ylabel('DNA Concentration [ng/µL]', fontsize=12)
                ax2.set_title(f'Experiment {exp_id}: DNA Concentration', fontsize=14)
                ax2.set_yscale('log')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Format x-axis
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Process Step', fontsize=12)

                plt.tight_layout()

                # Save figure
                exp_dir = output_dir / "experiment_details"
                exp_dir.mkdir(exist_ok=True)
                fig_path = exp_dir / f"experiment_{exp_id}_process_steps.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        except Exception as e:
            print(f"  Warning: Could not create detail plot for experiment {exp_id}: {e}")

    print(f"  All plots saved to {output_dir}")


def main():
    """Main function to control analysis workflow."""
    parser = argparse.ArgumentParser(description='Run cell lysis analysis')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to input data file (default: data/scFv/scfv_lysis.xlsx)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory (default: results/analysis)')
    parser.add_argument('--loocv', action='store_true',
                        help='Run Leave-One-Out Cross-Validation')
    parser.add_argument('--standard', action='store_true',
                        help='Run standard analysis on full dataset')
    args = parser.parse_args()

    # Set paths
    data_path = args.data
    if data_path is not None:
        data_path = Path(data_path)
    else:
        data_path = DATA_DIR / DATA_FILENAME

    output_path = args.output
    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = RESULTS_DIR / "analysis"

    # Make sure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default to running both analyses if none specified
    if not args.loocv and not args.standard:
        args.loocv = True
        args.standard = True

    start_time = time.time()
    print("=== Cell Lysis Analysis Pipeline ===")

    try:
        # Load data
        print(f"\nLoading data from {data_path}...")
        df_raw = load_data(data_path)
        print(f"Loaded {len(df_raw)} rows of data")

        # Run requested analyses
        if args.standard:
            print("\n=== Running Standard Analysis ===")
            standard_output = output_path / "standard"
            run_standard_analysis(df_raw, standard_output)

        if args.loocv:
            print("\n=== Running LOOCV Analysis ===")
            loocv_output = output_path / "loocv"
            run_loocv(df_raw, loocv_output)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n=== Analysis complete ({end_time - start_time:.2f} seconds) ===")


if __name__ == "__main__":
    main()