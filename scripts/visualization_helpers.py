#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import from existing codebase
from src.analysis.visualization import VisualizationManager


def generate_plots(df_raw, df_intact_pred, df_dna_pred, intact_model, dna_model, output_dir):
    """
    Generate extensive plots for both models using improved styling.

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

    # 3. Grid plot of intact fraction vs process step
    viz.plot_intact_vs_process_step(df_intact_pred, title="Intact Fraction vs Process Steps")

    # 4. Yield contour plots
    viz.plot_yield_contour(intact_model.params["k"], intact_model.params["alpha"], F0=1.0,
                          subdir="contour_plots")

    # If frozen biomass exists, create separate contour plot
    if "frozen biomass" in df_raw["biomass_type"].str.lower().unique():
        frozen_data = df_raw[df_raw["biomass_type"].str.lower() == "frozen biomass"].copy()
        if not frozen_data.empty and "intact_biomass_percent" in frozen_data.columns:
            frozen_F0 = frozen_data["intact_biomass_percent"].iloc[0] / 100.0
            if 0 < frozen_F0 <= 1:
                viz.plot_yield_contour(intact_model.params["k"], intact_model.params["alpha"], F0=frozen_F0,
                                      subdir="contour_plots")

    # 5. Residual plots
    viz.plot_residuals(df_intact_pred, 'observed_frac', 'intact_frac_pred',
                      title="Intact Fraction Residual Analysis")

    # --- DNA Model Plots ---
    print("  Creating DNA model plots...")

    # 1. DNA parity plots (both log and linear)
    viz.plot_dna_parity(df_dna_pred, title="DNA Concentration: Observed vs Predicted", log_scale=True)
    viz.plot_dna_parity(df_dna_pred, title="DNA Concentration: Observed vs Predicted", log_scale=False)

    # 2. NEW: Combined plot with intact fraction and DNA
    viz.plot_combined_intact_dna_vs_process(df_dna_pred, log_scale_dna=True,
                                          title="Intact Fraction and DNA Concentration")
    viz.plot_combined_intact_dna_vs_process(df_dna_pred, log_scale_dna=False,
                                          title="Intact Fraction and DNA Concentration")

    # 3. DNA residual plots (both log and linear)
    viz.plot_residuals(df_dna_pred, 'dna_conc', 'dna_pred',
                      title="DNA Concentration Residual Analysis", log_scale=True)
    viz.plot_residuals(df_dna_pred, 'dna_conc', 'dna_pred',
                      title="DNA Concentration Residual Analysis", log_scale=False)

    print(f"  All plots saved to {output_dir}")


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
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
