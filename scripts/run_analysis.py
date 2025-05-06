# scripts/run_analysis.py
import logging
import pandas as pd
from pathlib import Path
import sys
import time
import argparse
import json

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scFv"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# --- Setup logging first ---
def setup_logging(debug=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format
    )

    # Create file handler to also log to file
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Add file handler to root logger
    logging.getLogger('').addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


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
    # from analysis.models.two_compartment_dna_model import TwoCompartmentDNAModel
    from analysis.models.two_compartment_dna_model import TwoCompartmentMechanisticModel
    from analysis.models.simplified_compartment_model import SimplifiedCompartmentModel

    # Import cross-validation
    from analysis.cross_validation import CrossValidator

    # Import evaluation and unified visualization
    from analysis.evaluation import calculate_metrics, calculate_dna_metrics
    from analysis.unified_visualization import VisualizationManager
except ImportError as e:
    logging.error(f"Error importing analysis modules: {e}")
    logging.error(f"Attempted to import from: {SRC_DIR}")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during imports: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
DATA_FILENAME = "scfv_lysis.xlsx"

# Define all available model types
AVAILABLE_MODEL_TYPES = [
    # "single_wash",
    # "step_dependent_wash",
    # "continuous_decay",
    # "concentration_dependent",
    # "saturation",
    # "physical_adsorption",
    "two_compartment",
    # "simplified_compartment"
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
        # "single_wash": SingleWashDNAModel,
        # "step_dependent_wash": StepDependentWashDNAModel,
        # "continuous_decay": ContinuousDecayDNAModel,
        # "concentration_dependent": ConcentrationDependentDNAModel,
        # "saturation": SaturationDNAModel,
        # "physical_adsorption": PhysicalAdsorptionDNAModel,
        "two_compartment": TwoCompartmentMechanisticModel,
        # "simplified_compartment": SimplifiedCompartmentModel
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
    logging.info("=== Sensitivity Analysis ===")

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
            "W_base": [0.3, 0.4, 0.5, 0.6, 0.7],
            "lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
            "W_min": [0.05, 0.1, 0.2, 0.3, 0.4],
            "W_max": [0.6, 0.7, 0.8, 0.9, 0.95],
            "beta": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "W_inf": [0.05, 0.1, 0.2, 0.3, 0.4],
            "W_0": [0.6, 0.7, 0.8, 0.9, 0.95]
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
        logging.info("\nAnalyzing sensitivity for %s model", model_type)

        # Create base models
        intact_model = IntactModel()
        dna_model = create_dna_model(model_type)

        # Get parameters relevant for this model
        params_to_vary = get_model_parameter_names(model_type)

        # First, fit the base models to get reference values
        logging.info("Fitting base models...")
        intact_model.fit(df_intact_fit)
        df_with_intact_pred = intact_model.predict(df_raw)
        dna_model.fit(df_with_intact_pred)

        base_metrics = calculate_dna_metrics(dna_model.predict(df_with_intact_pred))
        base_params = {**intact_model.get_params(), **dna_model.get_params()}

        logging.info("Base model metrics: %s", base_metrics)
        logging.debug("Base model parameters: %s", base_params)

        # Store results for this model type
        model_results = []

        # Now vary each parameter individually
        for param in params_to_vary:
            if param not in param_ranges:
                continue

            logging.info("\nVarying parameter: %s", param)

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

                logging.info("  %s = %s: R² = %s, RMSE = %s",
                             param, value, metrics.get('R²', 'N/A'), metrics.get('RMSE', 'N/A'))

            # Create individual parameter sensitivity plots using VisualizationManager
            try:
                viz.plot_parameter_sensitivity(param_results,
                                               target_params=[param],
                                               target_metrics=["R²", "RMSE", "MAE"],
                                               subdir=model_type)
                logging.info("  Generated sensitivity plot for %s", param)
            except Exception as e:
                logging.error("  Error generating sensitivity plot for %s: %s", param, e)

        # Generate combined visualization for this model type using VisualizationManager
        model_subdirectory = f"{model_type}"
        viz.plot_parameter_sensitivity(model_results, subdir=model_subdirectory)

    # Save sensitivity analysis results
    results_path = results_dir / "sensitivity_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info("Sensitivity analysis results saved to %s", results_path)

    # Generate cross-model comparison plots
    logging.info("\nGenerating cross-model comparison plots...")
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
    logging.info("\n=== Running Leave-One-Out Cross-Validation ===")

    # Use all available models if none specified
    if model_types is None:
        model_types = AVAILABLE_MODEL_TYPES

    # Verify data before proceeding
    logging.info("Verifying data before LOOCV:")

    # Check required columns
    required_cols = ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step',
                     'total_passages_650', 'total_passages_1000', 'intact_biomass_percent']

    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        return {}

    # Check for NaN values in critical columns
    for col in required_cols:
        nan_count = df_raw[col].isna().sum()
        logging.info("Column '%s' has %d/%d NaN values (%.1f%%)",
                     col, nan_count, len(df_raw), nan_count / len(df_raw) * 100)

    # Check experiment IDs
    valid_exp_ids = df_raw.dropna(subset=['experiment_id'])
    exp_ids = valid_exp_ids['experiment_id'].unique()
    logging.info("Found %d unique experiment IDs: %s", len(exp_ids), exp_ids)

    if len(exp_ids) < 2:
        logging.error("Need at least 2 different experiment IDs for LOOCV")
        return {}

    # Add observed_frac
    if 'observed_frac' not in df_raw.columns:
        logging.info("Adding 'observed_frac' column from 'intact_biomass_percent'")
        df_raw['observed_frac'] = df_raw['intact_biomass_percent'] / 100.0

    # Check for enough valid data points
    valid_data = (df_raw['observed_frac'].notna() &
                  (df_raw['total_passages_650'].notna() | df_raw['total_passages_1000'].notna()))

    logging.info("Found %d/%d valid data points for modeling", valid_data.sum(), len(df_raw))

    if valid_data.sum() < 4:  # At least 4 points for meaningful LOOCV
        logging.error("Not enough valid data points for LOOCV")
        return {}

    # Filter data for intact model fitting
    df_intact_fit = filter_df_for_modeling(df_raw)

    # Create results directory
    results_dir = RESULTS_DIR / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization manager for overall results
    main_viz = VisualizationManager(results_dir)

    loocv_results = {}

    # Run LOOCV for each model type
    for model_type in model_types:
        logging.info("\n--- Running LOOCV for %s Model ---", model_type.replace('_', ' ').title())

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
            logging.error("%s model LOOCV failed to produce valid results", model_type)
            model_results = {"error": "LOOCV failed", "overall_metrics_dna": {}}

        loocv_results[model_type] = model_results

    # Compare Results
    logging.info("\n=== LOOCV Model Comparison ===")

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
    logging.info("\nComparison of DNA model performance:")
    logging.info("%-25s %-10s %-10s %-10s %-10s", "Model", "R²", "RMSE", "MAE", "RMSLE")
    logging.info("-" * 65)

    for model, metrics in comparison.items():
        model_name = model.replace('_', ' ').title()
        r2 = f"{metrics['R²']:.4f}" if isinstance(metrics['R²'], (int, float)) and not pd.isna(metrics['R²']) else "N/A"
        rmse = f"{metrics['RMSE']:.2f}" if isinstance(metrics['RMSE'], (int, float)) and not pd.isna(
            metrics['RMSE']) else "N/A"
        mae = f"{metrics['MAE']:.2f}" if isinstance(metrics['MAE'], (int, float)) and not pd.isna(
            metrics['MAE']) else "N/A"
        rmsle = f"{metrics['RMSLE']:.4f}" if isinstance(metrics['RMSLE'], (int, float)) and not pd.isna(
            metrics['RMSLE']) else "N/A"
        logging.info("%-25s %-10s %-10s %-10s %-10s", model_name, r2, rmse, mae, rmsle)

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

        logging.info("Model comparison saved to %s", comparison_path)

        # Generate additional visualizations only if we have valid metrics
        try:
            logging.info("--- Generating Additional Visualizations ---")
            logging.info("Creating model comparison chart...")
            main_viz.create_comparison_chart(comparison, "DNA Model Performance Comparison")

            # Generate additional comparative visualizations
            logging.info("Creating model residual plots...")
            for model_type in model_types:
                model_dir = results_dir / f"dna_model_{model_type}"
                if model_dir.exists():
                    # Load the predictions
                    pred_file = model_dir / "dna_loocv_predictions.csv"
                    if pred_file.exists():
                        try:
                            df_preds = pd.read_csv(pred_file)
                            model_viz = VisualizationManager(model_dir)
                            # Plot residuals
                            model_viz.plot_residuals(df_preds, 'dna_conc', 'dna_pred',
                                                     title=f"{model_type.replace('_', ' ').title()} Residuals",
                                                     log_scale=True)
                            logging.info("Created residual plots for %s model", model_type)
                        except Exception as e:
                            logging.error("Error creating residual plots for %s: %s", model_type, e)
        except Exception as e:
            logging.error("Error generating comparison visualizations: %s", e)
    else:
        logging.info("Skipping model comparison visualization - no valid metrics available")

    return loocv_results


def run_full_model_analysis(df_raw, model_type="step_dependent_wash"):
    """
    Run a full analysis using a specific model type.

    Args:
        df_raw: Raw data DataFrame
        model_type: Model type to use (default: step_dependent_wash)

    Returns:
        Dictionary with analysis results
    """
    logging.info("\n=== Running Full Model Analysis with %s Model ===", model_type.replace('_', ' ').title())

    # Create results directory
    results_dir = RESULTS_DIR / f"full_model_{model_type}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization manager
    viz = VisualizationManager(results_dir)

    # Filter data for intact model fitting
    df_intact_fit = filter_df_for_modeling(df_raw)

    # 1. Fit intact model
    logging.info("Fitting intact fraction model...")
    intact_model = IntactModel()
    intact_success = intact_model.fit(df_intact_fit)

    if not intact_success:
        logging.error("Failed to fit intact model.")
        return {"error": "Failed to fit intact model"}

    # Get intact model parameters
    k = intact_model.params["k"]
    alpha = intact_model.params["alpha"]
    logging.info("Fitted intact model parameters: k=%.3e, alpha=%.3f", k, alpha)

    # 2. Generate intact predictions
    logging.info("Generating intact fraction predictions...")
    df_with_intact_pred = intact_model.predict(df_raw)

    # 3. Calculate metrics for intact model
    intact_metrics = calculate_metrics(df_with_intact_pred)
    logging.info("Intact model metrics: %s", intact_metrics)

    # 4. Create intact model plots
    logging.info("Generating intact model plots...")

    # Intact parity plot
    viz.plot_intact_parity(df_with_intact_pred)

    # Overview fitted vs dose plot
    viz.plot_overview_fitted(df_with_intact_pred, k, alpha)

    # Yield contour plots for different biomass types
    viz.plot_yield_contour(k, alpha, F0=1.0, subdir="contour_plots")  # Fresh biomass

    # If frozen biomass exists, create separate contour plot
    if "frozen biomass" in df_raw["biomass_type"].str.lower().unique():
        # Find a representative F0 value for frozen biomass
        frozen_data = df_raw[df_raw["biomass_type"].str.lower() == "frozen biomass"].copy()
        if not frozen_data.empty:
            if "intact_biomass_percent" in frozen_data.columns:
                frozen_F0 = frozen_data["intact_biomass_percent"].iloc[0] / 100.0
                if 0 < frozen_F0 <= 1:
                    viz.plot_yield_contour(k, alpha, F0=frozen_F0, subdir="contour_plots")

    # 5. Fit DNA model
    logging.info("Fitting DNA model (%s)...", model_type)
    dna_model = create_dna_model(model_type)
    dna_success = dna_model.fit(df_with_intact_pred)

    if not dna_success:
        logging.error("Failed to fit DNA model.")
        return {
            "intact_model": {
                "parameters": intact_model.params,
                "metrics": intact_metrics
            },
            "error": "Failed to fit DNA model"
        }

    # Get DNA model parameters
    dna_params = dna_model.params
    logging.info("Fitted DNA model parameters: %s", dna_params)

    # 6. Generate DNA predictions
    logging.info("Generating DNA predictions...")
    df_with_dna_pred = dna_model.predict(df_with_intact_pred)

    # 7. Calculate metrics for DNA model
    dna_metrics = calculate_dna_metrics(df_with_dna_pred)
    logging.info("DNA model metrics: %s", dna_metrics)

    # 8. Create DNA model plots
    logging.info("Generating DNA model plots...")

    # DNA parity plot
    viz.plot_dna_parity(df_with_dna_pred)

    # DNA vs process step plot
    viz.plot_dna_vs_step(df_with_dna_pred, config_name=model_type)

    # Residual plots
    viz.plot_residuals(df_with_dna_pred, 'dna_conc', 'dna_pred', log_scale=True)

    # 9. Save predictions and results
    pred_path = results_dir / "predictions.csv"
    df_with_dna_pred.to_csv(pred_path, index=False)
    logging.info("Saved predictions to %s", pred_path)

    # 10. Compile and return results
    results = {
        "intact_model": {
            "parameters": intact_model.params,
            "metrics": intact_metrics
        },
        "dna_model": {
            "model_type": model_type,
            "parameters": dna_model.params,
            "metrics": dna_metrics
        },
        "prediction_path": str(pred_path)
    }

    # Save results to JSON
    results_path = results_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info("Saved results to %s", results_path)

    return results


def main():
    """Main function to control analysis workflow."""
    parser = argparse.ArgumentParser(description='Run cell lysis analysis with various options')
    parser.add_argument('--no-loocv', action='store_true', help='Skip LOOCV analysis')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specify which models to run (e.g., single_wash continuous_decay)')
    parser.add_argument('--full-model', action='store_true',
                        help='Run full model analysis with best model')
    parser.add_argument('--model-type', default='step_dependent_wash',
                        help='Model type to use for full model analysis')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed error information')
    args = parser.parse_args()

    # Set up logging first thing
    log_file = setup_logging(debug=args.debug)

    # If models are specified, validate them
    if args.models:
        invalid_models = [m for m in args.models if m not in AVAILABLE_MODEL_TYPES]
        if invalid_models:
            logging.error(f"Invalid model types: {invalid_models}")
            logging.error(f"Valid model types are: {AVAILABLE_MODEL_TYPES}")
            sys.exit(1)

    start_time = time.time()

    logging.info("=== Cell Lysis Analysis Pipeline ===")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Results directory: {RESULTS_DIR}")

    if args.models:
        logging.info(f"Selected models: {args.models}")
    else:
        logging.info(f"Using all available models: {AVAILABLE_MODEL_TYPES}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    logging.info("\n[1] Loading data...")
    try:
        df_raw = load_data(DATA_DIR / DATA_FILENAME)
        logging.info(f"Loaded {len(df_raw)} raw rows.")

        if df_raw.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # Add diagnostic information about data
        if args.debug:
            logging.debug("Data Summary:")
            logging.debug(f"Shape: {df_raw.shape}")
            logging.debug(f"Columns: {df_raw.columns.tolist()}")
            logging.debug("Data Types:\n%s", df_raw.dtypes)
            logging.debug("NaN Values by Column:")
            for col in df_raw.columns:
                nan_count = df_raw[col].isna().sum()
                logging.debug(f"{col}: {nan_count} NaN values ({nan_count / len(df_raw) * 100:.1f}%)")

            logging.debug("Unique Values for Categorical Columns:")
            for col in ['biomass_type', 'wash_procedure', 'process_step', 'experiment_id']:
                if col in df_raw.columns:
                    unique_values = df_raw[col].dropna().unique()
                    logging.debug(f"{col}: {unique_values}")

    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_DIR / DATA_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during data loading: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # --- Run Full Model Analysis if requested ---
    if args.full_model:
        try:
            model_type = args.model_type
            if model_type not in AVAILABLE_MODEL_TYPES:
                logging.error(f"Invalid model type: {model_type}")
                logging.error(f"Using default model type: step_dependent_wash")
                model_type = "step_dependent_wash"

            run_full_model_analysis(df_raw, model_type=model_type)
        except Exception as e:
            logging.exception(f"Error during full model analysis: {e}")

    # --- Run LOOCV if not disabled ---
    if not args.no_loocv:
        try:
            run_cross_validation(df_raw, model_types=args.models)
        except Exception as e:
            logging.exception(f"Error during cross-validation: {e}")

    # --- Run Sensitivity Analysis if requested ---
    if args.sensitivity:
        try:
            run_sensitivity_analysis(df_raw, model_types=args.models)
        except Exception as e:
            logging.exception(f"Error during sensitivity analysis: {e}")

    # --- Finish ---
    end_time = time.time()
    logging.info(f"\n=== Analysis complete ({end_time - start_time:.2f} seconds) ===")
    print(f"Analysis complete! Log file: {log_file}")


if __name__ == "__main__":
    main()