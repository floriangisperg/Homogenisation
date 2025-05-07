# scripts/residual_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import necessary modules
from analysis.visualization import VisualizationManager
from analysis.evaluation import calculate_dna_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# MODEL TYPES - Add or remove models as needed
MODEL_TYPES = [
    "single_wash",
    "step_dependent_wash",
    "continuous_decay",
    "concentration_dependent",
    "saturation",
    "physical_adsorption",
    "two_compartment",
    "simplified_compartment"
]


def find_prediction_files():
    """Automatically find all prediction files in the results directory."""
    found_files = {}

    # Look in full_model_* directories
    for model_type in MODEL_TYPES:
        model_dir = PROJECT_ROOT / "results" / f"full_model_{model_type}"
        pred_file = model_dir / "predictions.csv"
        if pred_file.exists():
            found_files[model_type] = pred_file
            print(f"Found predictions for '{model_type}' model: {pred_file}")

    # Also look in cross_validation directories
    cv_dir = PROJECT_ROOT / "results" / "cross_validation"
    if cv_dir.exists():
        for model_type in MODEL_TYPES:
            model_cv_dir = cv_dir / f"dna_model_{model_type}"
            if model_cv_dir.exists():
                cv_pred_file = model_cv_dir / "dna_loocv_predictions.csv"
                if cv_pred_file.exists():
                    key = f"{model_type}_cv"
                    found_files[key] = cv_pred_file
                    print(f"Found CV predictions for '{model_type}' model: {cv_pred_file}")

    if not found_files:
        print("No prediction files found!")

    return found_files


def load_predictions(predictions_file):
    """Load prediction results from CSV file."""
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return None

    try:
        df = pd.read_csv(predictions_file)
        print(f"Loaded {len(df)} rows from {predictions_file}")

        required_cols = ['dna_conc', 'dna_pred']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing required columns: {missing}")
            return None

        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None


def calculate_residuals(df):
    """Add residual and relative residual columns to the DataFrame."""
    df = df.copy()

    # Calculate absolute residuals
    df['residual'] = df['dna_conc'] - df['dna_pred']

    # Calculate relative (percentage) residuals
    valid_obs = df['dna_conc'] > 0
    df['rel_residual'] = np.nan
    df.loc[valid_obs, 'rel_residual'] = (df.loc[valid_obs, 'residual'] /
                                         df.loc[valid_obs, 'dna_conc'] * 100)

    # Calculate log residuals
    valid_both = (df['dna_conc'] > 0) & (df['dna_pred'] > 0)
    df['log_residual'] = np.nan
    df.loc[valid_both, 'log_residual'] = (np.log10(df.loc[valid_both, 'dna_conc']) -
                                          np.log10(df.loc[valid_both, 'dna_pred']))

    return df


def create_heatmap_matrix(df, output_dir):
    """Create a heatmap showing prediction errors by experiment and process step."""
    # Filter to valid data points
    valid_data = df['dna_conc'].notna() & df['dna_pred'].notna() & (df['dna_conc'] > 0)
    df_valid = df[valid_data].copy()

    if df_valid.empty or 'experiment_id' not in df_valid.columns or 'process_step' not in df_valid.columns:
        print("Cannot create heatmap: missing data or required columns.")
        return

    print("Creating prediction error heatmap...")

    # Calculate relative error
    df_valid['rel_error'] = np.abs(df_valid['dna_conc'] - df_valid['dna_pred']) / df_valid['dna_conc'] * 100

    # Create a pivot table of mean relative error by experiment and process step
    pivot_error = df_valid.pivot_table(
        values='rel_error',
        index='experiment_id',
        columns='process_step',
        aggfunc='mean'
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_error, annot=True, cmap='YlOrRd', fmt='.1f', ax=ax)

    ax.set_title('Mean Relative Error (%) by Experiment and Process Step')
    plt.tight_layout()

    plot_file = output_dir / "error_heatmap_experiment_step.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved error heatmap to {plot_file}")

    # Also create a heatmap by biomass type and process step
    if 'biomass_type' in df_valid.columns:
        pivot_biomass_step = df_valid.pivot_table(
            values='rel_error',
            index='biomass_type',
            columns='process_step',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_biomass_step, annot=True, cmap='YlOrRd', fmt='.1f', ax=ax)

        ax.set_title('Mean Relative Error (%) by Biomass Type and Process Step')
        plt.tight_layout()

        plot_file = output_dir / "error_heatmap_biomass_step.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved biomass-step error heatmap to {plot_file}")


def analyze_residuals_by_group(df, group_col, output_dir):
    """Analyze residuals broken down by a grouping variable."""
    if group_col not in df.columns:
        print(f"Column {group_col} not found. Skipping group analysis.")
        return

    # Filter to valid data points
    valid_data = (df['dna_conc'] > 0) & (df['dna_pred'] > 0) & df['dna_conc'].notna() & df['dna_pred'].notna()
    df_valid = df[valid_data].copy()

    if df_valid.empty:
        print("No valid data points for analysis.")
        return

    # Get unique values of the grouping variable
    groups = df_valid[group_col].unique()
    n_groups = len(groups)

    if n_groups == 0:
        print(f"No groups found for {group_col}.")
        return

    print(f"Analyzing residuals by {group_col} ({n_groups} groups)...")

    # Create metrics table by group
    metrics_by_group = {}

    for group_val in groups:
        group_data = df_valid[df_valid[group_col] == group_val]

        if not group_data.empty:
            metrics = calculate_dna_metrics(group_data)
            metrics_by_group[group_val] = {
                'count': len(group_data),
                'R²': metrics.get('R²', np.nan),
                'RMSE': metrics.get('RMSE', np.nan),
                'MAPE': metrics.get('MAPE', np.nan),
                'mean_residual': group_data['residual'].mean(),
                'mean_rel_residual': group_data['rel_residual'].mean(),
            }

    # Create metrics table as DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics_by_group, orient='index')
    metrics_df.index.name = group_col
    metrics_df.reset_index(inplace=True)

    # Save metrics table
    metrics_file = output_dir / f"metrics_by_{group_col}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics by {group_col} to {metrics_file}")

    # Create boxplot of residuals by group
    plt.figure(figsize=(12, 6))

    # Relative residuals (capped for better visualization)
    rel_residual_plot = df_valid[(df_valid['rel_residual'] > -200) & (df_valid['rel_residual'] < 200)]
    sns.boxplot(x=group_col, y='rel_residual', data=rel_residual_plot)
    plt.title(f'Percentage Residuals by {group_col}')
    plt.ylabel('Percentage Residual (%)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_file = output_dir / f"residuals_boxplot_by_{group_col}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved residuals boxplot to {plot_file}")

    return metrics_df


def run_residual_analysis():
    """Run comprehensive residual analysis on all found model predictions."""
    prediction_files = find_prediction_files()

    if not prediction_files:
        print("No prediction files found to analyze!")
        return

    # Create main output directory
    output_dir = PROJECT_ROOT / "results" / "residual_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each found file
    for model_name, pred_file in prediction_files.items():
        print(f"\n=== Analyzing {model_name} model ===")

        # Create model-specific output directory
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        # Load predictions
        df = load_predictions(pred_file)
        if df is None:
            continue

        # Calculate residuals
        df = calculate_residuals(df)

        # Create basic plots
        try:
            # Create a visualization manager for basic plots
            viz = VisualizationManager(model_output_dir)

            # Generate standard residual plots
            print("Generating standard residual plots...")
            viz.plot_residuals(df, 'dna_conc', 'dna_pred', log_scale=True,
                               title=f"{model_name} Residual Analysis")
        except Exception as e:
            print(f"Error generating standard plots: {e}")

        # Try to create a parity plot using the visualization manager
        try:
            print("Generating parity plot...")
            viz.plot_dna_parity(df, title=f"{model_name} Model")
        except Exception as e:
            print(f"Error generating parity plot: {e}")

            # Fallback: Create a simple parity plot
            try:
                valid_data = (df['dna_conc'] > 0) & (df['dna_pred'] > 0)
                plt.figure(figsize=(8, 8))
                plt.scatter(df.loc[valid_data, 'dna_pred'], df.loc[valid_data, 'dna_conc'], alpha=0.7)
                plt.xscale('log')
                plt.yscale('log')
                min_val = min(df.loc[valid_data, 'dna_pred'].min(), df.loc[valid_data, 'dna_conc'].min()) * 0.8
                max_val = max(df.loc[valid_data, 'dna_pred'].max(), df.loc[valid_data, 'dna_conc'].max()) * 1.2
                plt.plot([min_val, max_val], [min_val, max_val], 'k--')
                plt.xlabel('Predicted DNA [ng/μL]')
                plt.ylabel('Observed DNA [ng/μL]')
                plt.title(f'{model_name} Model: Observed vs Predicted')
                plt.savefig(model_output_dir / 'simple_parity_plot.png', dpi=300)
                plt.close()
                print("Created simple parity plot as fallback")
            except Exception as e:
                print(f"Error creating fallback parity plot: {e}")

        # Analyze residuals by group
        try:
            for group in ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step']:
                if group in df.columns:
                    analyze_residuals_by_group(df, group, model_output_dir)
        except Exception as e:
            print(f"Error analyzing residuals by group: {e}")

        # Create heatmap matrices
        try:
            create_heatmap_matrix(df, model_output_dir)
        except Exception as e:
            print(f"Error creating heatmap: {e}")

        # Generate summary statistics
        try:
            overall_metrics = calculate_dna_metrics(df)

            # Save summary
            summary_file = model_output_dir / "analysis_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Residual Analysis Summary for {model_name} Model\n")
                f.write("=" * 50 + "\n\n")

                f.write("Overall Metrics:\n")
                for k, v in overall_metrics.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

                valid_count = sum((df['dna_conc'] > 0) & (df['dna_pred'] > 0))
                f.write(f"Total samples: {len(df)}\n")
                f.write(f"Valid samples: {valid_count} ({valid_count / len(df) * 100:.1f}%)\n")

            print(f"Saved summary to {summary_file}")
        except Exception as e:
            print(f"Error generating summary: {e}")

    print("\nResidual analysis complete!")


if __name__ == "__main__":
    print("Starting residual analysis...")
    # Just run the function directly - no arguments needed
    run_residual_analysis()