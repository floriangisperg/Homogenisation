#!/usr/bin/env python3
# scripts/run_biomass_specific_loocv.py
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
from analysis.models.biomass_specific_dna_model import BiomassSpecificDNAModel
from analysis.evaluation import calculate_metrics, calculate_dna_metrics
from analysis.visualization import VisualizationManager

# --- Constants ---
DATA_FILENAME = "scfv_lysis.xlsx"


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


def plot_parameter_distributions(fold_parameters, output_dir):
    """
    Create visualization of parameter distributions across folds

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
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)


def run_biomass_specific_loocv(df_raw, output_dir=None, regularization_lambda=0.1):
    """
    Run Leave-One-Out Cross-Validation with biomass-specific DNA model.

    Args:
        df_raw: Raw data DataFrame
        output_dir: Path to the output directory (optional)
        regularization_lambda: Regularization strength for biomass-specific parameters

    Returns:
        Dictionary with enhanced LOOCV results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "biomass_specific_loocv"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Biomass-Specific Leave-One-Out Cross-Validation")
    print(f"Saving results to {output_dir}")
    print(f"Using regularization lambda: {regularization_lambda}")

    # Handle NA values in experiment_id before sorting
    df_valid = df_raw.dropna(subset=['experiment_id']).copy()
    exp_ids = sorted([int(x) for x in df_valid['experiment_id'].unique() if pd.notna(x)])
    n_folds = len(exp_ids)

    if n_folds < 2:
        print(f"Error: Need at least 2 experiments for LOOCV. Found {n_folds}.")
        return {"error": "Insufficient experiments for LOOCV"}

    # Check that we have both biomass types
    biomass_types = df_valid['biomass_type'].unique()
    has_fresh = any(bt == 'fresh biomass' for bt in biomass_types)
    has_frozen = any(bt == 'frozen biomass' for bt in biomass_types)

    if not (has_fresh and has_frozen):
        print(f"Warning: Need both biomass types for biomass-specific model. Found: {biomass_types}")
        print("Consider using regular model instead.")

    # Count experiments by biomass type
    frozen_exps = []
    fresh_exps = []

    for exp_id in exp_ids:
        exp_data = df_valid[df_valid['experiment_id'] == exp_id]
        if exp_data.empty:
            continue

        biomass_type = exp_data['biomass_type'].iloc[0]
        if biomass_type == 'frozen biomass':
            frozen_exps.append(exp_id)
        elif biomass_type == 'fresh biomass':
            fresh_exps.append(exp_id)

    print(f"Found {len(frozen_exps)} frozen biomass experiments: {frozen_exps}")
    print(f"Found {len(fresh_exps)} fresh biomass experiments: {fresh_exps}")

    if len(frozen_exps) < 2 or len(fresh_exps) < 2:
        print("Warning: Less than 2 experiments for at least one biomass type.")
        print("Cross-validation may not be representative for all biomass types.")

    print(f"Running {n_folds} LOOCV folds with experiments: {exp_ids}")

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

        # Get biomass type of test experiment
        test_biomass = test_df['biomass_type'].iloc[0]
        print(f"Test experiment biomass type: {test_biomass}")

        # Check if we still have both biomass types in training set
        train_biomass_types = train_df['biomass_type'].unique()
        if len(train_biomass_types) < 2:
            print(f"Warning: Training set only has biomass types: {train_biomass_types}")
            print("Biomass-specific model may not be trained optimally.")

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
            "test_biomass": test_biomass,
            "k": intact_model.params["k"],
            "alpha": intact_model.params["alpha"]
        })

        # 2. Generate intact predictions for training data
        train_with_intact_pred = intact_model.predict(train_df)

        # 3. Fit DNA model on training data
        print(f"Fitting biomass-specific DNA model on training data...")
        dna_model = BiomassSpecificDNAModel()
        # Set regularization strength
        dna_model.regularization_lambda = regularization_lambda
        dna_success = dna_model.fit(train_with_intact_pred)

        if not dna_success:
            print(f"Failed to fit DNA model for fold {fold_num}. Skipping.")
            continue

        # Save parameters
        fold_parameters["dna"].append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            "test_biomass": test_biomass,
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
            "test_biomass": test_biomass,
            **{k: v for k, v in intact_metrics.items() if k != "N_valid"}
        })

        fold_dna_metrics_table.append({
            "fold": fold_num,
            "test_exp_id": test_exp_id,
            "test_biomass": test_biomass,
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
            "test_biomass": test_biomass,
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

    # Calculate metrics by biomass type
    print("\nCalculating metrics by biomass type...")

    # For frozen biomass
    frozen_preds = all_preds_df[all_preds_df['biomass_type'] == 'frozen biomass'].copy()
    if not frozen_preds.empty:
        frozen_intact_metrics = calculate_metrics(frozen_preds)
        frozen_dna_metrics = calculate_dna_metrics(frozen_preds)
        print(f"Frozen biomass metrics:")
        print(
            f"  Intact: R²={frozen_intact_metrics.get('R²', 'N/A'):.4f}, RMSE={frozen_intact_metrics.get('RMSE', 'N/A'):.4f}")
        print(f"  DNA: R²={frozen_dna_metrics.get('R²', 'N/A'):.4f}, RMSE={frozen_dna_metrics.get('RMSE', 'N/A'):.1f}")
        if 'NRMSE_mean' in frozen_dna_metrics and pd.notna(frozen_dna_metrics['NRMSE_mean']):
            print(f"  DNA normalized: NRMSE(mean)={frozen_dna_metrics.get('NRMSE_mean', 'N/A'):.3f}")
    else:
        frozen_intact_metrics = {}
        frozen_dna_metrics = {}
        print("No valid frozen biomass predictions.")

    # For fresh biomass
    fresh_preds = all_preds_df[all_preds_df['biomass_type'] == 'fresh biomass'].copy()
    if not fresh_preds.empty:
        fresh_intact_metrics = calculate_metrics(fresh_preds)
        fresh_dna_metrics = calculate_dna_metrics(fresh_preds)
        print(f"Fresh biomass metrics:")
        print(
            f"  Intact: R²={fresh_intact_metrics.get('R²', 'N/A'):.4f}, RMSE={fresh_intact_metrics.get('RMSE', 'N/A'):.4f}")
        print(f"  DNA: R²={fresh_dna_metrics.get('R²', 'N/A'):.4f}, RMSE={fresh_dna_metrics.get('RMSE', 'N/A'):.1f}")
        if 'NRMSE_mean' in fresh_dna_metrics and pd.notna(fresh_dna_metrics['NRMSE_mean']):
            print(f"  DNA normalized: NRMSE(mean)={fresh_dna_metrics.get('NRMSE_mean', 'N/A'):.3f}")
    else:
        fresh_intact_metrics = {}
        fresh_dna_metrics = {}
        print("No valid fresh biomass predictions.")

    # Print overall metrics including normalized metrics
    print(f"\nOverall LOOCV metrics:")
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

        avg_params[model_type] = {
            "overall": {},
            "by_biomass_type": {
                "frozen": {},
                "fresh": {}
            }
        }

        # Calculate overall averages
        for param in fold_parameters[model_type][0].keys():
            if param in ["fold", "test_exp_id", "test_biomass"]:
                continue

            values = [fold[param] for fold in fold_parameters[model_type] if param in fold]
            if values:
                avg_params[model_type]["overall"][param] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }

        # Calculate averages by biomass type (based on test set biomass)
        frozen_folds = [fold for fold in fold_parameters[model_type] if fold.get("test_biomass") == "frozen biomass"]
        fresh_folds = [fold for fold in fold_parameters[model_type] if fold.get("test_biomass") == "fresh biomass"]

        if frozen_folds:
            for param in frozen_folds[0].keys():
                if param in ["fold", "test_exp_id", "test_biomass"]:
                    continue

                values = [fold[param] for fold in frozen_folds if param in fold]
                if values:
                    avg_params[model_type]["by_biomass_type"]["frozen"][param] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }

        if fresh_folds:
            for param in fresh_folds[0].keys():
                if param in ["fold", "test_exp_id", "test_biomass"]:
                    continue

                values = [fold[param] for fold in fresh_folds if param in fold]
                if values:
                    avg_params[model_type]["by_biomass_type"]["fresh"][param] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }

    # Generate overall plots
    print("\nGenerating overall LOOCV plots...")
    viz = VisualizationManager(output_dir)

    # Intact parity plot for all experiments
    viz.plot_intact_parity(all_preds_df, title="Biomass-Specific LOOCV: Intact Fraction")

    # DNA parity plot for all experiments
    viz.plot_dna_parity(all_preds_df, title="Biomass-Specific LOOCV: DNA Concentration")

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
        "biomass_specific_metrics": {
            "frozen": {
                "intact": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in frozen_intact_metrics.items()},
                "dna": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in frozen_dna_metrics.items()}
            },
            "fresh": {
                "intact": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in fresh_intact_metrics.items()},
                "dna": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in fresh_dna_metrics.items()}
            }
        },
        "average_parameters": avg_params,
        "regularization_lambda": regularization_lambda
    }

    results_path = output_dir / "biomass_specific_loocv_results.json"
    with open(results_path, 'w') as f:
        json.dump(loocv_results, f, indent=4)
    print(f"Saved LOOCV results to {results_path}")

    # Plot parameter distributions
    try:
        plot_parameter_distributions(fold_parameters, output_dir)
    except Exception as e:
        print(f"Warning: Could not plot parameter distributions: {e}")

    return loocv_results


def main():
    """Main function to run biomass-specific LOOCV analysis."""
    parser = argparse.ArgumentParser(description='Run cell lysis analysis with biomass-specific DNA model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to input data file (default: data/scFv/scfv_lysis.xlsx)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory (default: results/biomass_specific_loocv)')
    parser.add_argument('--reg', type=float, default=0.1,
                        help='Regularization lambda to control overfitting (default: 0.1)')
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
        output_path = RESULTS_DIR / "biomass_specific_loocv"

    # Make sure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print("=== Biomass-Specific Cell Lysis Analysis Pipeline ===")

    try:
        # Load data
        print(f"\nLoading data from {data_path}...")
        df_raw = load_data(data_path)
        print(f"Loaded {len(df_raw)} rows of data")

        # Run biomass-specific LOOCV
        run_biomass_specific_loocv(df_raw, output_path, regularization_lambda=args.reg)

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