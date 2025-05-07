# scripts/symbolic_regression_dna.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Try to import gplearn (PySR would be another option but requires Julia)
try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    print("gplearn not installed. Installing now...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "gplearn"])
    from gplearn.genetic import SymbolicRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths configuration
DATA_DIR = PROJECT_ROOT / "data" / "scFv"
DATA_FILENAME = "scfv_lysis.xlsx"
RESULTS_DIR = PROJECT_ROOT / "results" / "symbolic_regression"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load and preprocess data for symbolic regression."""
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Standardize column names
    column_mapping = {
        "Process step": "process_step",
        "total number of passages at 650 bar": "total_passages_650",
        "total number of passages at 1000 bar": "total_passages_1000",
        "Intact biomass percentage [%]": "intact_biomass_percent",
        "DNA [ng/µL]": "dna_conc",
        "DNA std. dev. [ng/µL]": "dna_std_dev",
        "wash procedure": "wash_procedure",
        "biomass type": "biomass_type",
        "experiment id": "experiment_id"
    }

    # Rename columns that exist in the DataFrame
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=columns_to_rename)

    # Convert numeric columns
    numeric_cols = ["total_passages_650", "total_passages_1000", "intact_biomass_percent",
                    "dna_conc", "dna_std_dev"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Process categorical columns
    categorical_cols = ["process_step", "wash_procedure", "biomass_type"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().str.lower()

    # Calculate observed_frac
    if 'intact_biomass_percent' in df.columns:
        df["observed_frac"] = df["intact_biomass_percent"] / 100.0

    # Create numeric representations for categorical variables
    if 'biomass_type' in df.columns:
        df["is_frozen"] = (df["biomass_type"] == "frozen biomass").astype(int)

    if 'wash_procedure' in df.columns:
        df["is_recursive"] = (df["wash_procedure"] == "recursive wash").astype(int)

    # Create step number features
    if 'process_step' in df.columns:
        # Map process steps to numeric values
        step_mapping = {
            "resuspended biomass": 0,
            "initial lysis": 1,
            "1st wash": 2,
            "2nd wash": 3,
            "3rd wash": 4,
            "4th wash": 5
        }
        df["step_number"] = df["process_step"].map(step_mapping)

        # Create a binary feature for indicating wash steps
        df["is_wash_step"] = df["process_step"].str.contains("wash").astype(int)

    # Calculate cumulative passages
    df["total_passages"] = df["total_passages_650"] + df["total_passages_1000"]

    # Filter to only rows with DNA data
    df_with_dna = df.dropna(subset=["dna_conc"]).copy()

    logging.info(f"Preprocessed data: {len(df_with_dna)} rows with DNA data")

    return df_with_dna


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for symbolic regression."""
    # Define features to use
    feature_cols = [
        "total_passages_650",
        "total_passages_1000",
        "intact_biomass_percent",
        "is_frozen",
        "is_recursive",
        "step_number",
        "observed_frac",
        "total_passages"
    ]

    # Ensure we have all needed columns
    features = [col for col in feature_cols if col in df.columns]

    # Create feature matrix, filling NaNs with 0
    X = df[features].fillna(0)

    # Define target
    target = df["dna_conc"]

    return X, target


def run_loocv_symbolic_regression(df: pd.DataFrame,
                                  population_size=1000,
                                  generations=100) -> Dict[str, Any]:
    """
    Run Leave-One-Out Cross-Validation with Symbolic Regression.

    Args:
        df: DataFrame with DNA data
        population_size: Population size for genetic programming
        generations: Number of generations to run

    Returns:
        Dictionary with LOOCV results
    """
    logging.info("Starting LOOCV with Symbolic Regression...")

    # Get unique experiment IDs
    exp_ids = df["experiment_id"].unique()
    n_folds = len(exp_ids)

    if n_folds < 2:
        logging.error("Need at least 2 experiments for LOOCV")
        return {"error": "Insufficient experiments for LOOCV"}

    # Initialize results containers
    fold_results = []
    all_programs = []
    best_programs = []
    predictions = []

    # Run LOOCV
    for i, held_out_id in enumerate(exp_ids):
        logging.info(f"Fold {i + 1}/{n_folds}: Holding out Experiment {held_out_id}")

        # Split data
        train_data = df[df["experiment_id"] != held_out_id].copy()
        test_data = df[df["experiment_id"] == held_out_id].copy()

        if train_data.empty or test_data.empty:
            logging.warning(f"Skipping fold {i + 1}: Empty train or test set")
            continue

        # Prepare features
        X_train, y_train = prepare_features(train_data)
        X_test, y_test = prepare_features(test_data)

        # Create subset of features with best correlation to target
        # This helps symbolic regression focus on the most relevant features
        correlations = {}
        for col in X_train.columns:
            correlations[col] = abs(X_train[col].corr(y_train))

        # Sort features by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        feature_subset = [f[0] for f in sorted_features[:6]]  # Use top 6 features

        X_train_subset = X_train[feature_subset]
        X_test_subset = X_test[feature_subset]

        # Set up symbolic regression
        symbolic_regressor = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv'],
            metric='mse',
            const_range=(-10.0, 10.0),
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=0.01,
            random_state=42,
            n_jobs=-1,  # Use all cores
            verbose=1
        )

        # Run symbolic regression
        logging.info(f"Running symbolic regression for fold {i + 1}...")
        start_time = time.time()

        # Fit the model
        symbolic_regressor.fit(X_train_subset.values, y_train.values)

        # Get the best program
        best_program = symbolic_regressor._program
        all_programs.append(symbolic_regressor)
        best_programs.append(best_program)

        # Make predictions
        y_pred = symbolic_regressor.predict(X_test_subset.values)

        # Calculate metrics
        metrics = {
            "R²": r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        }

        # Save results for this fold
        fold_result = {
            "fold": i + 1,
            "experiment_id": held_out_id,
            "equation": str(best_program),
            "feature_subset": feature_subset,
            "metrics": metrics,
            "runtime": time.time() - start_time
        }

        fold_results.append(fold_result)

        # Save test predictions
        test_data["dna_pred"] = y_pred
        predictions.append(test_data)

        logging.info(f"Fold {i + 1} complete: R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.2f}")
        logging.info(f"Best equation: {best_program}")

    # Combine all predictions
    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    # Calculate overall metrics
    overall_metrics = {}
    if not all_predictions.empty:
        overall_metrics = {
            "R²": r2_score(all_predictions["dna_conc"], all_predictions["dna_pred"]),
            "RMSE": np.sqrt(mean_squared_error(all_predictions["dna_conc"], all_predictions["dna_pred"])),
            "MAE": mean_absolute_error(all_predictions["dna_conc"], all_predictions["dna_pred"])
        }

    # Compile results
    loocv_results = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "overall_metrics": overall_metrics,
        "predictions": all_predictions
    }

    return loocv_results


def visualize_results(results: Dict[str, Any]) -> None:
    """Visualize the results of symbolic regression LOOCV."""
    if "error" in results:
        logging.error(f"Cannot visualize results: {results['error']}")
        return

    if "predictions" not in results or results["predictions"].empty:
        logging.error("No predictions available to visualize")
        return

    predictions = results["predictions"]

    # 1. Create parity plot
    plt.figure(figsize=(8, 8))
    valid_idx = (predictions["dna_conc"] > 0) & (predictions["dna_pred"] > 0)
    plt.scatter(predictions.loc[valid_idx, "dna_conc"],
                predictions.loc[valid_idx, "dna_pred"],
                c=predictions.loc[valid_idx, "experiment_id"],
                alpha=0.7)

    # Add y=x line
    min_val = min(predictions.loc[valid_idx, "dna_conc"].min(),
                  predictions.loc[valid_idx, "dna_pred"].min()) * 0.8
    max_val = max(predictions.loc[valid_idx, "dna_conc"].max(),
                  predictions.loc[valid_idx, "dna_pred"].max()) * 1.2
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Observed DNA Concentration [ng/µL]')
    plt.ylabel('Predicted DNA Concentration [ng/µL]')
    plt.title('Symbolic Regression LOOCV: Observed vs Predicted')

    # Add R² value
    r2 = results["overall_metrics"].get("R²", np.nan)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Save plot
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "symbolic_regression_parity_plot.png", dpi=300)
    plt.close()

    # 2. Create plot of predictions vs process step for each experiment
    exp_ids = predictions["experiment_id"].unique()
    for exp_id in exp_ids:
        exp_data = predictions[predictions["experiment_id"] == exp_id].copy()
        if exp_data.empty or "step_number" not in exp_data.columns:
            continue

        plt.figure(figsize=(10, 6))

        # Sort by step number
        exp_data = exp_data.sort_values("step_number")

        # Plot observed and predicted
        plt.plot(exp_data["step_number"], exp_data["dna_conc"], 'o-', label='Observed')
        plt.plot(exp_data["step_number"], exp_data["dna_pred"], 'x--', label='Predicted')

        # Set log scale for y-axis
        plt.yscale('log')

        # Get experiment info
        biomass_type = exp_data["biomass_type"].iloc[0] if "biomass_type" in exp_data else "unknown"
        wash_procedure = exp_data["wash_procedure"].iloc[0] if "wash_procedure" in exp_data else "unknown"

        # Add step names to x-axis
        if "process_step" in exp_data:
            plt.xticks(exp_data["step_number"], exp_data["process_step"], rotation=45, ha='right')

        # Labels and title
        plt.xlabel('Process Step')
        plt.ylabel('DNA Concentration [ng/µL]')
        plt.title(f'Experiment {exp_id}: {biomass_type}, {wash_procedure}')
        plt.legend()

        # Add grid
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

        # Save plot
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"symbolic_regression_exp_{exp_id}.png", dpi=300)
        plt.close()


def save_results(results: Dict[str, Any]) -> None:
    """Save the results of symbolic regression LOOCV."""
    if "error" in results:
        logging.error(f"Cannot save results: {results['error']}")
        return

    # Save predictions to CSV
    if "predictions" in results and not results["predictions"].empty:
        predictions_file = RESULTS_DIR / "symbolic_regression_predictions.csv"
        results["predictions"].to_csv(predictions_file, index=False)
        logging.info(f"Saved predictions to {predictions_file}")

    # Save fold results
    if "fold_results" in results and results["fold_results"]:
        # Extract key information for the summary
        summary_rows = []
        for fold in results["fold_results"]:
            summary_rows.append({
                "Experiment": fold["experiment_id"],
                "R²": fold["metrics"]["R²"],
                "RMSE": fold["metrics"]["RMSE"],
                "MAE": fold["metrics"]["MAE"],
                "Features": ", ".join(fold.get("feature_subset", [])),
                "Equation": fold["equation"]
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_file = RESULTS_DIR / "symbolic_regression_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Saved summary to {summary_file}")

    # Save detailed report
    report_file = RESULTS_DIR / "symbolic_regression_report.txt"
    with open(report_file, 'w') as f:
        f.write("Symbolic Regression LOOCV Results\n")
        f.write("=================================\n\n")

        # Overall metrics
        f.write("Overall Metrics:\n")
        if "overall_metrics" in results:
            for metric, value in results["overall_metrics"].items():
                f.write(f"  {metric}: {value:.6f}\n")

        # Individual fold results
        if "fold_results" in results and results["fold_results"]:
            f.write("\nIndividual Fold Results:\n")
            for fold in results["fold_results"]:
                f.write(f"\nExperiment {fold['experiment_id']}:\n")
                f.write(f"  R²: {fold['metrics']['R²']:.6f}\n")
                f.write(f"  RMSE: {fold['metrics']['RMSE']:.6f}\n")
                f.write(f"  MAE: {fold['metrics']['MAE']:.6f}\n")
                f.write(f"  Features used: {', '.join(fold.get('feature_subset', []))}\n")
                f.write(f"  Equation: {fold['equation']}\n")

    logging.info(f"Saved detailed report to {report_file}")


def main():
    """Main function to run symbolic regression with LOOCV."""
    logging.info("=== Symbolic Regression with LOOCV for DNA Concentration Modeling ===")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_data(DATA_DIR / DATA_FILENAME)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Run LOOCV with symbolic regression
    results = run_loocv_symbolic_regression(df, population_size=1000, generations=100)

    # Visualize results
    visualize_results(results)

    # Save results
    save_results(results)

    logging.info("=== Symbolic Regression Analysis Complete ===")


if __name__ == "__main__":
    main()