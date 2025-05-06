# scripts/run_analysis.py
import pandas as pd
from pathlib import Path
import sys
import time
import argparse
import json

from matplotlib import pyplot as plt

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scFv"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# --- Imports ---
try:
    from analysis.data_processing import load_data, filter_df_for_modeling

    # Import model classes
    from analysis.models.intact_model import IntactModel
    from analysis.models.dna_models import SingleWashDNAModel, StepDependentWashDNAModel
    from analysis.models.alternative_dna_models import (
        ContinuousDecayDNAModel,
        ConcentrationDependentDNAModel,
        SaturationDNAModel,
        PhysicalAdsorptionDNAModel
    )

    # Import cross-validation
    from analysis.cross_validation import CrossValidator

    # Import evaluation and visualization
    from analysis.evaluation import calculate_metrics, calculate_dna_metrics
    from analysis.visualization import VisualizationManager
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print(f"Attempted to import from: {SRC_DIR}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
DATA_FILENAME = "scfv_lysis.xlsx"

# Define all available model types
AVAILABLE_MODEL_TYPES = [
    "single_wash",
    "step_dependent_wash",
    "continuous_decay",
    "concentration_dependent",
    "saturation",
    "physical_adsorption"
]


# --- Helper Functions ---
def create_dna_model(model_type):
    """
    Factory function to create DNA model instances based on model type.

    Args:
        model_type: String identifying the model type

    Returns:
        An instance of the specified DNA model
    """
    model_map = {
        "single_wash": SingleWashDNAModel,
        "step_dependent_wash": StepDependentWashDNAModel,
        "continuous_decay": ContinuousDecayDNAModel,
        "concentration_dependent": ConcentrationDependentDNAModel,
        "saturation": SaturationDNAModel,
        "physical_adsorption": PhysicalAdsorptionDNAModel
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_map.keys())}")

    return model_map[model_type]()


def get_model_parameter_names(model_type):
    """
    Returns the parameter names for a given model type.

    Args:
        model_type: String identifying the model type

    Returns:
        List of parameter names that can be varied for the model
    """
    # Common parameters for all models
    common_params = ["k", "alpha", "C_release_fresh", "C_release_frozen"]

    model_specific_params = {
        "single_wash": ["W_wash"],
        "step_dependent_wash": ["W_wash_1st", "W_wash_subsequent"],
        "continuous_decay": ["W_base", "lambda"],
        "concentration_dependent": ["W_min", "W_max", "beta"],
        "saturation": ["W_min", "W_max", "lambda"],
        "physical_adsorption": ["W_inf", "W_0", "k"]
    }

    return common_params + model_specific_params.get(model_type, [])


# --- Main functions ---
def run_sensitivity_analysis(df_raw, model_types=None, param_ranges=None):
    """
    Run a sensitivity analysis by varying model parameters.

    Args:
        df_raw: Raw data DataFrame
        model_types: List of model types to test
        param_ranges: Dictionary with parameter ranges to test

    Returns:
        List of dictionaries with sensitivity analysis results
    """
    print("=== Sensitivity Analysis ===")

    if model_types is None:
        model_types = AVAILABLE_MODEL_TYPES

    if param_ranges is None:
        # Default parameter ranges to test
        param_ranges = {
            "k": [1e-7, 5e-7, 1e-6, 5e-6, 1e-5],
            "alpha": [1.0, 1.25, 1.5, 1.75, 2.0],
            "C_release_fresh": [15000, 20000, 25000],
            "C_release_frozen": [15000, 20000, 25000],
            "W_wash": [0.3, 0.4, 0.5, 0.6, 0.7],
            "W_wash_1st": [0.3, 0.4, 0.5, 0.6, 0.7],
            "W_wash_subsequent": [0.3, 0.4, 0.5, 0.6, 0.7],
            # Parameters for continuous decay model
            "W_base": [0.3, 0.4, 0.5, 0.6, 0.7],
            "lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
            # Parameters for concentration dependent model
            "W_min": [0.05, 0.1, 0.2, 0.3, 0.4],
            "W_max": [0.6, 0.7, 0.8, 0.9, 0.95],
            "beta": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            # Parameters for physical adsorption model
            "W_inf": [0.05, 0.1, 0.2, 0.3, 0.4],
            "W_0": [0.6, 0.7, 0.8, 0.9, 0.95]
            # Note: physical adsorption also uses "k" which is already defined
        }

    # Filter data for intact model fitting
    df_intact_fit = filter_df_for_modeling(df_raw)

    # Create results directory
    results_dir = RESULTS_DIR / "sensitivity_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization manager
    viz = VisualizationManager(results_dir)

    # Store overall results
    results = []

    # For simplicity, let's just vary one parameter at a time
    for model_type in model_types:
        print(f"\nAnalyzing sensitivity for {model_type} model")

        # Create base models
        intact_model = IntactModel()
        dna_model = create_dna_model(model_type)

        # Get parameters relevant for this model
        params_to_vary = get_model_parameter_names(model_type)

        # First, fit the base models to get reference values
        print("Fitting base models...")
        intact_model.fit(df_intact_fit)
        df_with_intact_pred = intact_model.predict(df_raw)
        dna_model.fit(df_with_intact_pred)

        base_metrics = calculate_dna_metrics(dna_model.predict(df_with_intact_pred))
        base_params = {**intact_model.get_params(), **dna_model.get_params()}

        print(f"Base model metrics: {base_metrics}")
        print(f"Base model parameters: {base_params}")

        # Store results for this model type
        model_results = []

        # Now vary each parameter individually
        for param in params_to_vary:
            if param not in param_ranges:
                continue

            print(f"\nVarying parameter: {param}")

            # Add the base value to the range if not already included
            if param in base_params and base_params[param] not in param_ranges[param]:
                param_values = sorted(param_ranges[param] + [base_params[param]])
            else:
                param_values = sorted(param_ranges[param])

            # Store results for this parameter
            param_results = []

            for value in param_values:
                # Create new models with base parameters
                test_intact_model = IntactModel()
                test_intact_model.set_params(intact_model.get_params())

                test_dna_model = create_dna_model(model_type)
                test_dna_model.set_params(dna_model.get_params())

                # Set the varied parameter
                if param in test_intact_model.params:
                    test_intact_model.params[param] = value
                elif param in test_dna_model.params:
                    test_dna_model.params[param] = value

                # Generate predictions
                df_test_intact_pred = test_intact_model.predict(df_raw)
                df_test_final_pred = test_dna_model.predict(df_test_intact_pred)

                # Calculate metrics
                metrics = calculate_dna_metrics(df_test_final_pred)

                result = {
                    "model_type": model_type,
                    "param_name": param,
                    "param_value": value,
                    "base_value": base_params.get(param, "N/A"),
                    "metrics": metrics
                }

                param_results.append(result)
                model_results.append(result)
                results.append(result)

                print(f"  {param} = {value}: R² = {metrics.get('R²', 'N/A')}, "
                      f"RMSE = {metrics.get('RMSE', 'N/A')}")

            # Create individual parameter sensitivity plot
            param_data = {
                model_type: {
                    "R²": [r["metrics"].get("R²") for r in param_results],
                    "RMSE": [r["metrics"].get("RMSE") for r in param_results],
                    "MAE": [r["metrics"].get("MAE") for r in param_results]
                }
            }

            x_values = [r["param_value"] for r in param_results]

            # Plot each metric separately
            metrics_to_plot = ["R²", "RMSE", "MAE"]

            for metric in metrics_to_plot:
                fig_title = f"{model_type.replace('_', ' ').title()} - {param} vs {metric}"
                subdir = f"{model_type}"

                # Create figure
                plt.figure(figsize=(8, 6))
                plt.plot(x_values, param_data[model_type][metric], marker='o', linestyle='-',
                         label=metric)

                # Mark the base value
                base_value = base_params.get(param)
                if base_value in x_values:
                    idx = x_values.index(base_value)
                    metric_value = param_data[model_type][metric][idx]
                    plt.plot(base_value, metric_value, marker='*', markersize=12, color='red',
                             label=f"Base Value: {base_value}")

                plt.xlabel(f"{param}")
                plt.ylabel(metric)
                plt.title(fig_title)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()

                # Save the figure
                filename = f"{model_type}_{param}_{metric.replace('²', '2')}.png"
                save_path = results_dir / subdir
                save_path.mkdir(parents=True, exist_ok=True)

                try:
                    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
                    print(f"  Saved sensitivity plot to {save_path / filename}")
                except Exception as e:
                    print(f"  Error saving sensitivity plot: {e}")

                plt.close()

        # Generate combined visualization for this model type
        model_subdirectory = f"{model_type}"
        viz.plot_parameter_sensitivity(model_results, subdir=model_subdirectory)

    # Save sensitivity analysis results
    results_path = results_dir / "sensitivity_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nSensitivity analysis results saved to {results_path}")

    # Generate cross-model comparison plots
    print("\nGenerating cross-model comparison plots...")
    viz.plot_parameter_sensitivity(results, target_params=["k", "alpha"],
                                   target_metrics=["R²", "RMSE"])

    return results


def run_cross_validation(df_raw, model_types=None):
    """
    Run Leave-One-Out Cross-Validation for all model types.

    Args:
        df_raw: Raw data DataFrame
        model_types: List of model types to test

    Returns:
        Dictionary with LOOCV results for all models
    """
    print("\n=== Running Leave-One-Out Cross-Validation ===")

    # Use all available models if none specified
    if model_types is None:
        model_types = AVAILABLE_MODEL_TYPES

    # --- Verify data before proceeding ---
    print("\nVerifying data before LOOCV:")

    # Check required columns
    required_cols = ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step',
                     'total_passages_650', 'total_passages_1000', 'intact_biomass_percent']

    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return {}

    # Check for NaN values in critical columns
    for col in required_cols:
        nan_count = df_raw[col].isna().sum()
        print(f"  Column '{col}' has {nan_count}/{len(df_raw)} NaN values ({nan_count / len(df_raw) * 100:.1f}%)")

    # Check experiment IDs
    valid_exp_ids = df_raw.dropna(subset=['experiment_id'])
    exp_ids = valid_exp_ids['experiment_id'].unique()
    print(f"  Found {len(exp_ids)} unique experiment IDs: {exp_ids}")

    if len(exp_ids) < 2:
        print("ERROR: Need at least 2 different experiment IDs for LOOCV")
        return {}

    # Add observed_frac
    if 'observed_frac' not in df_raw.columns:
        print("  Adding 'observed_frac' column from 'intact_biomass_percent'")
        df_raw['observed_frac'] = df_raw['intact_biomass_percent'] / 100.0

    # Check for enough valid data points
    valid_data = (df_raw['observed_frac'].notna() &
                  (df_raw['total_passages_650'].notna() | df_raw['total_passages_1000'].notna()))

    print(f"  Found {valid_data.sum()}/{len(df_raw)} valid data points for modeling")

    if valid_data.sum() < 4:  # At least 4 points for meaningful LOOCV
        print("ERROR: Not enough valid data points for LOOCV")
        return {}

    # Filter data for intact model fitting
    df_intact_fit = filter_df_for_modeling(df_raw)

    # Create results directory
    results_dir = RESULTS_DIR / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization manager
    viz = VisualizationManager(results_dir)

    loocv_results = {}

    # --- Run LOOCV for each model type ---
    for model_type in model_types:
        print(f"\n--- Running LOOCV for {model_type.replace('_', ' ').title()} Model ---")

        # Create the CrossValidator with the appropriate model class
        model_cv = CrossValidator(
            output_dir=results_dir,
            intact_model_class=IntactModel,
            dna_model_class=lambda: create_dna_model(model_type)
        )

        model_results = model_cv.run_loocv(
            df_raw=df_raw,
            df_intact_fit_base=df_intact_fit,
            model_variant=model_type
        )

        # Check if we got valid results
        if model_results is None or "error" in model_results:
            print(f"ERROR: {model_type} model LOOCV failed to produce valid results")
            model_results = {"error": "LOOCV failed", "overall_metrics_dna": {}}

        loocv_results[model_type] = model_results

    # --- Compare Results ---
    print("\n=== LOOCV Model Comparison ===")

    comparison = {}
    for model_type in model_types:
        metrics = loocv_results[model_type].get("overall_metrics_dna", {})
        comparison[model_type] = {
            "R²": metrics.get("R²", float('nan')),
            "RMSE": metrics.get("RMSE", float('nan')),
            "MAE": metrics.get("MAE", float('nan')),
            "RMSLE": metrics.get("RMSLE", float('nan')),
            "MAPE": metrics.get("MAPE", float('nan')),
            "Bias": metrics.get("Bias", float('nan'))
        }

    # Print comparison table
    print("\nComparison of DNA model performance:")
    print(f"{'Model':<25} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'RMSLE':<10}")
    print("-" * 65)

    for model, metrics in comparison.items():
        model_name = model.replace('_', ' ').title()
        r2 = f"{metrics['R²']:.4f}" if isinstance(metrics['R²'], (int, float)) and not pd.isna(metrics['R²']) else "N/A"
        rmse = f"{metrics['RMSE']:.2f}" if isinstance(metrics['RMSE'], (int, float)) and not pd.isna(
            metrics['RMSE']) else "N/A"
        mae = f"{metrics['MAE']:.2f}" if isinstance(metrics['MAE'], (int, float)) and not pd.isna(
            metrics['MAE']) else "N/A"
        rmsle = f"{metrics['RMSLE']:.4f}" if isinstance(metrics['RMSLE'], (int, float)) and not pd.isna(
            metrics['RMSLE']) else "N/A"
        print(f"{model_name:<25} {r2:<10} {rmse:<10} {mae:<10} {rmsle:<10}")

    # Save comparison only if there are valid metrics
    has_valid_metrics = False
    for _, metrics in comparison.items():
        if any(not pd.isna(v) for v in metrics.values()):
            has_valid_metrics = True
            break

    if has_valid_metrics:
        # Convert NaN values to None for JSON serialization
        for model in comparison:
            for metric in comparison[model]:
                if isinstance(comparison[model][metric], float) and pd.isna(comparison[model][metric]):
                    comparison[model][metric] = None

        comparison_path = results_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=4)

        print(f"\nModel comparison saved to {comparison_path}")

        # Generate additional visualizations only if we have valid metrics
        try:
            print("\n--- Generating Additional Visualizations ---")
            print("  Creating model comparison chart...")
            viz.create_comparison_chart(comparison, "DNA Model Performance Comparison")
        except Exception as e:
            print(f"Error generating comparison chart: {e}")
    else:
        print("\nSkipping model comparison visualization - no valid metrics available")

    return loocv_results


def main():
    """Main function to control analysis workflow."""
    parser = argparse.ArgumentParser(description='Run cell lysis analysis with various options')
    parser.add_argument('--no-loocv', action='store_true', help='Skip LOOCV analysis')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specify which models to run (e.g., single_wash continuous_decay)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed error information')
    args = parser.parse_args()

    # If models are specified, validate them
    if args.models:
        invalid_models = [m for m in args.models if m not in AVAILABLE_MODEL_TYPES]
        if invalid_models:
            print(f"Error: Invalid model types: {invalid_models}")
            print(f"Valid model types are: {AVAILABLE_MODEL_TYPES}")
            sys.exit(1)

    start_time = time.time()

    print("=== Cell Lysis Analysis Pipeline ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    if args.models:
        print(f"Selected models: {args.models}")
    else:
        print(f"Using all available models: {AVAILABLE_MODEL_TYPES}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    print("\n[1] Loading data...")
    try:
        df_raw = load_data(DATA_DIR / DATA_FILENAME)
        print(f"  Loaded {len(df_raw)} raw rows.")

        if df_raw.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # Add diagnostic information about data
        if args.debug:
            print("\nData Summary:")
            print(f"  Shape: {df_raw.shape}")
            print(f"  Columns: {df_raw.columns.tolist()}")
            print(f"  Data Types:\n{df_raw.dtypes}")
            print("\nNaN Values by Column:")
            for col in df_raw.columns:
                nan_count = df_raw[col].isna().sum()
                print(f"  {col}: {nan_count} NaN values ({nan_count / len(df_raw) * 100:.1f}%)")

            print("\nUnique Values for Categorical Columns:")
            for col in ['biomass_type', 'wash_procedure', 'process_step', 'experiment_id']:
                if col in df_raw.columns:
                    unique_values = df_raw[col].dropna().unique()
                    print(f"  {col}: {unique_values}")

            # Print first few rows for inspection
            print("\nFirst 5 rows of data:")
            print(df_raw.head())

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_DIR / DATA_FILENAME}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data loading: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # --- Run LOOCV ---
    if not args.no_loocv:
        try:
            run_cross_validation(df_raw, model_types=args.models)
        except Exception as e:
            print(f"Error during cross-validation: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # --- Run Sensitivity Analysis ---
    if args.sensitivity:
        try:
            run_sensitivity_analysis(df_raw, model_types=args.models)
        except Exception as e:
            print(f"Error during sensitivity analysis: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # --- Finish ---
    end_time = time.time()
    print(f"\n=== Analysis complete ({end_time - start_time:.2f} seconds) ===")


if __name__ == "__main__":
    main()