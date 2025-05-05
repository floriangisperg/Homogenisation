# scripts/run_analysis.py
import pandas as pd
from pathlib import Path
import sys
import time
import numpy as np
import json
import traceback
from scipy.optimize import minimize
from collections import defaultdict # To store fold metrics

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "scFv" # Ensure this is correct
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Add src directory to Python path
sys.path.insert(0, str(SRC_DIR))

# --- Imports from src/analysis ---
# (Keep existing imports)
try:
    from analysis.data_processing import (
        load_data, filter_df_for_modeling, add_cumulative_dose,
        calculate_delta_F, compute_cumulative_dose_values # Already imported
    )
    from analysis.optimization import (
        fit_intact_model,
        objective_dna_release,
        objective_dna_wash
    )
    from analysis.models.mechanistic_model import predict_intact_fraction
    from analysis.models.dna_model import predict_dna
    from analysis.evaluation import calculate_metrics, calculate_dna_metrics # Updated metrics calc
    from analysis.plotting import (
        plot_overview_fitted, plot_parity as plot_intact_parity,
        plot_yield_contour, plot_overview_observed_vs_dose,
        plot_overview_observed_vs_step,
        plot_dna_vs_step_matplotlib,
        plot_dna_parity_matplotlib
    )
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    # (Keep existing error handling)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
# (Keep existing constants)
DATA_FILENAME = "scfv_lysis.xlsx"
INITIAL_GUESS_INTACT = [1e-6, 1.5]
OPTIMIZATION_METHOD_INTACT = "Nelder-Mead"
OPTIMIZATION_METHOD_DNA = "Nelder-Mead"
INITIAL_GUESS_C_RELEASE = [20000]
INITIAL_GUESS_W_WASH = [0.5]
BOUNDS_W_WASH = [(0.001, 0.999)]

# --- Refactored Helper Functions for Fitting and Prediction ---

def fit_models(df_intact_fit_train: pd.DataFrame, df_raw_train: pd.DataFrame,
               initial_guess_intact: list, method_intact: str,
               initial_guess_C: list, initial_guess_W: list, method_dna: str, bounds_W: list):
    """
    Fits both intact and DNA models on the provided training data.

    Args:
        df_intact_fit_train: Filtered DataFrame for fitting k, alpha.
        df_raw_train: Raw DataFrame for preparing DNA fitting data.
        ... (other fitting parameters) ...

    Returns:
        A dictionary containing fitted parameters and success flags.
    """
    results = {
        "k": np.nan, "alpha": np.nan, "sse_fit_intact": np.nan, "success_intact": False,
        "C_release_fresh": np.nan, "C_release_frozen": np.nan, "W_wash": np.nan, "success_dna": False
    }

    # --- Stage 1: Fit Intact (k, alpha) ---
    if df_intact_fit_train.empty:
        print("  Skipping intact fit (empty training data).")
        return results # Return early, cannot proceed

    best_k, best_alpha, sse_intact, success_intact = fit_intact_model(
        df_intact_fit_train, initial_guess=initial_guess_intact, method=method_intact
    )
    results.update({"k": best_k, "alpha": best_alpha, "sse_fit_intact": sse_intact, "success_intact": success_intact})

    if not success_intact:
        print("  Intact fit failed on training data.")
        return results # Cannot proceed to DNA fit

    # --- Stage 2: Prepare Data for DNA Fitting (using training data) ---
    df_raw_train_copy = df_raw_train.copy()
    df_train_with_dose = add_cumulative_dose(df_raw_train_copy, best_k, best_alpha)
    df_train_intact_pred = predict_intact_fraction(df_train_with_dose, best_k, best_alpha)
    df_train_dna_base = calculate_delta_F(df_train_intact_pred)

    if 'dna_conc' not in df_train_dna_base.columns:
        print("  Error: 'dna_conc' missing in training data for DNA fit.")
        return results # Cannot proceed

    # --- Stage 3: Fit DNA Parameters (C_release, W_wash) ---
    best_C_fresh, best_C_frozen, best_W = np.nan, np.nan, np.nan

    # 3a: Fit C_release
    df_dna_train_release = df_train_dna_base[
        (df_train_dna_base["wash_procedure"] == "recursive wash") |
        ((df_train_dna_base["wash_procedure"] == "linear wash") & (df_train_dna_base["process_step"].str.lower() == "initial lysis"))
    ].copy()
    df_dna_train_release = df_dna_train_release[df_dna_train_release["process_step"].str.lower() != "resuspended biomass"]
    df_dna_train_release = df_dna_train_release.dropna(subset=["dna_conc", "delta_F"])

    # Fit C_frozen
    df_frozen_release_train = df_dna_train_release[df_dna_train_release["biomass_type"] == "frozen biomass"]
    if not df_frozen_release_train.empty and df_frozen_release_train['delta_F'].sum() > 1e-9:
        try:
            result_C_frozen = minimize(objective_dna_release, x0=initial_guess_C,
                                       args=(df_frozen_release_train, "frozen biomass"), method=method_dna)
            if result_C_frozen.success: best_C_frozen = max(0, result_C_frozen.x[0])
        except Exception as e: print(f"    Error fitting C_frozen on train: {e}")
    else: print("    Skipping C_frozen fit (no valid train data).")


    # Fit C_fresh
    df_fresh_release_train = df_dna_train_release[df_dna_train_release["biomass_type"] == "fresh biomass"]
    if not df_fresh_release_train.empty and df_fresh_release_train['delta_F'].sum() > 1e-9:
        try:
            result_C_fresh = minimize(objective_dna_release, x0=initial_guess_C,
                                      args=(df_fresh_release_train, "fresh biomass"), method=method_dna)
            if result_C_fresh.success: best_C_fresh = max(0, result_C_fresh.x[0])
        except Exception as e: print(f"    Error fitting C_fresh on train: {e}")
    else: print("    Skipping C_fresh fit (no valid train data).")


    # 3b: Fit W_wash
    df_linear_wash_decay_train_base = df_train_dna_base[
        (df_train_dna_base["wash_procedure"] == "linear wash") &
        (~df_train_dna_base["process_step"].str.lower().isin(["resuspended biomass", "initial lysis"]))
    ].copy()
    df_linear_wash_decay_train_base = df_linear_wash_decay_train_base.dropna(subset=["dna_conc"])

    dna_pred_release_map_train = {}
    for exp_id in df_linear_wash_decay_train_base["experiment_id"].unique():
        initial_lysis_row = df_train_dna_base[(df_train_dna_base["experiment_id"] == exp_id) & (df_train_dna_base["process_step"].str.lower() == "initial lysis")]
        if not initial_lysis_row.empty:
             delta_F_init = initial_lysis_row["delta_F"].iloc[0]
             is_frozen_init = initial_lysis_row["biomass_type"].iloc[0] == "frozen biomass"
             C_release_init = best_C_frozen if is_frozen_init else best_C_fresh
             if pd.notna(delta_F_init) and pd.notna(C_release_init): dna_pred_release_map_train[exp_id] = max(0, C_release_init * delta_F_init)
             else: dna_pred_release_map_train[exp_id] = np.nan
        else: dna_pred_release_map_train[exp_id] = np.nan

    df_linear_wash_decay_train = df_linear_wash_decay_train_base.copy()
    df_linear_wash_decay_train['dna_pred_release'] = df_linear_wash_decay_train['experiment_id'].map(dna_pred_release_map_train)
    df_linear_wash_decay_train = df_linear_wash_decay_train.dropna(subset=['dna_pred_release'])

    if not df_linear_wash_decay_train.empty:
        try:
             result_W = minimize(objective_dna_wash, x0=initial_guess_W, args=(df_linear_wash_decay_train,),
                                 method="L-BFGS-B", bounds=bounds_W)
             if result_W.success: best_W = result_W.x[0]
        except Exception as e: print(f"    Error fitting W_wash on train: {e}")
    else: print("    Skipping W_wash fit (no valid linear wash decay train data).")


    # --- Finalize DNA fit results ---
    # Success requires *both* C values (if applicable data exists) and W (if applicable data exists)
    # More robust check: Check if parameters are NaN based on whether data *existed* for them
    dna_data_exists = {
        "fresh": not df_fresh_release_train.empty,
        "frozen": not df_frozen_release_train.empty,
        "wash": not df_linear_wash_decay_train.empty
    }
    success_dna = (
        (not dna_data_exists["fresh"] or pd.notna(best_C_fresh)) and
        (not dna_data_exists["frozen"] or pd.notna(best_C_frozen)) and
        (not dna_data_exists["wash"] or pd.notna(best_W))
    )

    results.update({
        "C_release_fresh": best_C_fresh, "C_release_frozen": best_C_frozen, "W_wash": best_W,
        "success_dna": success_dna
    })

    return results


def predict_on_data(df_raw_target: pd.DataFrame, fit_results: dict) -> pd.DataFrame:
    """
    Generates predictions on target data using provided fitted parameters.

    Args:
        df_raw_target: Raw DataFrame to generate predictions for.
        fit_results: Dictionary containing fitted parameters (k, alpha, C_..., W_...).

    Returns:
        DataFrame with added 'intact_frac_pred' and 'dna_pred' columns.
    """
    df_target_pred = df_raw_target.copy()
    k = fit_results.get("k", np.nan)
    alpha = fit_results.get("alpha", np.nan)
    C_fresh = fit_results.get("C_release_fresh", np.nan)
    C_frozen = fit_results.get("C_release_frozen", np.nan)
    W = fit_results.get("W_wash", np.nan)

    # Check if intact fit succeeded before predicting
    if pd.isna(k) or pd.isna(alpha):
        print("  Skipping prediction (missing k or alpha).")
        df_target_pred["intact_frac_pred"] = np.nan
        df_target_pred["dna_pred"] = np.nan
        return df_target_pred

    # Predict Intact Fraction
    df_target_with_dose = add_cumulative_dose(df_target_pred, k, alpha)
    df_target_intact_pred = predict_intact_fraction(df_target_with_dose, k, alpha)

    # Prepare for DNA Prediction
    df_target_dna_base = calculate_delta_F(df_target_intact_pred)

    # Predict DNA (predict_dna should handle NaN parameters gracefully)
    df_final_predictions = predict_dna(df_target_dna_base, C_fresh, C_frozen, W)

    return df_final_predictions


# --- Helper Function for Running a Single Analysis (Original Version) ---
# (Keep the original run_single_analysis function as is for non-LOOCV runs)
def run_single_analysis(config_name: str, df_subset_fit_intact: pd.DataFrame, df_all_raw: pd.DataFrame, base_results_dir: Path):
    # ... (Keep the entire original function content here) ...
    # ... It will now use fit_models and predict_on_data internally ...
    # --- Refactor run_single_analysis slightly ---
    print(f"\n{'='*10} Starting Analysis for: {config_name} {'='*10}")
    results_subdir = base_results_dir / config_name
    results_subdir.mkdir(parents=True, exist_ok=True)
    print(f"  Results will be saved to: {results_subdir}")

    # --- Stage 1 & 3 Combined: Fit Models ---
    print(f"\n  [1+3] Fitting Intact and DNA Models using data for '{config_name}'...")
    fit_results = fit_models(
        df_subset_fit_intact, df_all_raw,
        INITIAL_GUESS_INTACT, OPTIMIZATION_METHOD_INTACT,
        INITIAL_GUESS_C_RELEASE, INITIAL_GUESS_W_WASH, OPTIMIZATION_METHOD_DNA, BOUNDS_W_WASH
    )

    # --- Stage 2 & 4 Combined: Generate Final Predictions ---
    print("\n  [2+4] Generating Final Predictions...")
    df_final_pred = predict_on_data(df_all_raw, fit_results) # Predict on the same data it was trained on

    # --- Stage 5: Evaluation ---
    print("\n  [5] Calculating Final Metrics...")
    # Intact metrics (calculate on the subset used for k/alpha fit)
    df_intact_fit_pred = predict_on_data(df_subset_fit_intact, fit_results) # Predict only on the fit subset
    intact_metrics = calculate_metrics(df_intact_fit_pred)
    print(f"  Intact Model Metrics ({config_name} data):", intact_metrics)

    # DNA metrics (calculate on all applicable points from the full raw data for this config)
    dna_metrics = calculate_dna_metrics(df_final_pred) # Use df_final_pred which has predictions for all raw points
    print(f"  DNA Model Metrics ({config_name} data):", dna_metrics)


    # --- Stage 6: Store Results ---
    final_summary = {
        "config_name": config_name,
        "k": fit_results["k"], "alpha": fit_results["alpha"],
        "C_release_fresh": fit_results["C_release_fresh"],
        "C_release_frozen": fit_results["C_release_frozen"],
        "W_wash": fit_results["W_wash"],
        "sse_fit_intact": fit_results["sse_fit_intact"], # SSE from the k/alpha fit
        "metrics_intact": intact_metrics,
        "metrics_dna": dna_metrics,
        "fit_successful_intact": fit_results["success_intact"],
        "fit_successful_dna": fit_results["success_dna"]
    }

    # --- Stage 7: Save Fit Summary ---
    summary_path = results_subdir / "fit_summary.json"
    try:
        # Create a serializable copy for JSON output
        serializable_results = final_summary.copy()
        # Convert metrics dicts (handle potential None values if metrics calculation failed)
        serializable_results['metrics_intact'] = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in (intact_metrics or {}).items()}
        serializable_results['metrics_dna'] = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in (dna_metrics or {}).items()}
        # Convert individual parameters, handling potential NaNs
        for key in ["k", "alpha", "C_release_fresh", "C_release_frozen", "W_wash", "sse_fit_intact"]:
             if key in serializable_results and pd.notna(serializable_results[key]):
                 serializable_results[key] = float(serializable_results[key])
             else:
                  serializable_results[key] = None # Store NaN as null in JSON

        with open(summary_path, 'w') as f:
             json.dump(serializable_results, f, indent=4, sort_keys=True)
        print(f"  Saved full fit summary to {summary_path}")
    except Exception as e:
        print(f"  Error saving full fit summary: {e}")


    # --- Stage 8: Generate Plots ---
    print("\n  [8] Generating Plots...")
    try:
        # Get unique experiment IDs present in the raw data for this config
        exp_ids_in_config = df_all_raw['experiment_id'].unique()
        exp_ids_in_config = [eid for eid in exp_ids_in_config if pd.notna(eid)]

        if len(exp_ids_in_config) == 0:
            print("  No valid experiments found in the raw data subset for this config, skipping plots.")
        else:
            # --- Intact Fraction Plots ---
            # Plot fitted overview (uses df_intact_fit_pred - data used for k/alpha fit + preds)
            if not df_intact_fit_pred.empty:
                plot_overview_fitted(df_intact_fit_pred, fit_results["k"], fit_results["alpha"], results_subdir)
                plot_intact_parity(df_intact_fit_pred, results_subdir) # Use only fit data for parity

            # Plot yield contour (uses k/alpha)
            if pd.notna(fit_results["k"]) and pd.notna(fit_results["alpha"]):
                plot_yield_contour(fit_results["k"], fit_results["alpha"], results_subdir, F0=1.0)
                 # Plot yield contour for average frozen F0 if possible
                frozen_raw_for_contour = df_all_raw[df_all_raw['biomass_type'] == 'frozen biomass']
                if not frozen_raw_for_contour.empty:
                     first_step_indices = frozen_raw_for_contour.loc[frozen_raw_for_contour.dropna(subset=['experiment_id']).groupby('experiment_id').apply(lambda x: x.index.min())].index
                     frozen_first_raw = frozen_raw_for_contour.loc[first_step_indices]
                     avg_F0_raw_frozen = (frozen_first_raw['intact_biomass_percent']/100.0).mean()
                     if pd.notna(avg_F0_raw_frozen) and 0 < avg_F0_raw_frozen <=1: plot_yield_contour(fit_results["k"], fit_results["alpha"], results_subdir, F0=avg_F0_raw_frozen)

            # Plot observed data overviews (use df_all_raw - all raw points for this config)
            # Ensure k, alpha are valid before passing to plotting functions needing them
            if pd.notna(fit_results["k"]) and pd.notna(fit_results["alpha"]):
                 plot_overview_observed_vs_dose(df_all_raw, fit_results["k"], fit_results["alpha"], results_subdir, config_name)
            else: print("Skipping plot_overview_observed_vs_dose due to invalid k/alpha.")
            plot_overview_observed_vs_step(df_all_raw, results_subdir, config_name)

            # --- DNA Plots ---
            # Use df_final_pred which contains observed DNA ('dna_conc') and predicted DNA ('dna_pred')
            if not df_final_pred.empty:
                plot_dna_vs_step_matplotlib(df_final_pred, results_subdir, config_name)
                plot_dna_parity_matplotlib(df_final_pred, results_subdir, config_name)

    except Exception as e:
         print(f"  ERROR during plotting for {config_name}: {e}")
         print("-" * 60); traceback.print_exc(); print("-" * 60)

    print(f"\n{'='*10} Finished Analysis for: {config_name} {'='*10}")
    return final_summary # Return the dictionary


# --- Main Execution Block ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Full Analysis Workflow (Intact + DNA) ---")

    # --- Initial Data Load and Preparation ---
    print(f"\n[A] Loading and preparing base data...")
    try:
        df_raw_full = load_data(DATA_DIR / DATA_FILENAME)
        print(f"  Loaded {len(df_raw_full)} raw rows.")
        if df_raw_full.empty: raise ValueError("Loaded DataFrame is empty.")
        df_model_base_intact = filter_df_for_modeling(df_raw_full)
        print(f"  Base dataset for INTACT model fitting contains {len(df_model_base_intact)} rows.")
        if df_model_base_intact.empty and not df_raw_full.empty:
             print("Warning: Filtering for intact model fitting resulted in empty DataFrame.")
    except FileNotFoundError:
         print(f"Error: Data file not found at {DATA_DIR / DATA_FILENAME}")
         sys.exit(1)
    except Exception as e:
        print(f"  Error during initial data load/prep: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Define Analysis Configurations ---
    # (Keep existing configurations)
    analysis_configs = [
        {'name': 'all_data', 'filter_lambda': lambda df: df},
        {'name': 'fresh_biomass', 'filter_lambda': lambda df: df[df['biomass_type'] == 'fresh biomass'].copy()},
        {'name': 'frozen_biomass', 'filter_lambda': lambda df: df[df['biomass_type'] == 'frozen biomass'].copy()}
    ]

    # --- Run Standard Analysis for Each Configuration ---
    all_fit_results = {}
    print("\n" + "="*80)
    print("--- Running Standard Fit-All Analysis ---")
    print("="*80)
    for config in analysis_configs:
        config_name = config['name']
        filter_func = config['filter_lambda']
        df_subset_for_intact_fitting = filter_func(df_model_base_intact)
        df_subset_raw_all_steps = filter_func(df_raw_full)

        # Pre-checks (keep existing checks)
        if df_subset_raw_all_steps.empty: print(f"\nSkipping config '{config_name}' (empty raw dataset)."); continue
        if 'experiment_id' not in df_subset_for_intact_fitting.columns:
            print(f"Critical Error: 'experiment_id' missing for config '{config_name}'. Skipping."); continue
        initial_rows = len(df_subset_for_intact_fitting)
        df_subset_for_intact_fitting = df_subset_for_intact_fitting.dropna(subset=['experiment_id'])
        if len(df_subset_for_intact_fitting) < initial_rows: print(f"  Info: Removed {initial_rows - len(df_subset_for_intact_fitting)} rows with missing 'experiment_id' for intact fitting in '{config_name}'.")
        if df_subset_for_intact_fitting.empty and config_name != 'all_data': print(f"\nSkipping config '{config_name}' (empty INTACT fitting dataset)."); continue

        # Run standard analysis
        results = run_single_analysis(
            config_name=config_name,
            df_subset_fit_intact=df_subset_for_intact_fitting,
            df_all_raw=df_subset_raw_all_steps,
            base_results_dir=RESULTS_DIR
        )
        if results: all_fit_results[config_name] = results

    # --- Run Leave-One-Out Cross-Validation (LOOCV) ---
    print("\n" + "="*80)
    print("--- Running Leave-One-Out Cross-Validation (LOOCV on all_data) ---")
    print("="*80)
    loocv_results_dir = RESULTS_DIR / "loocv_all_data"
    loocv_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"LOOCV results will be saved to: {loocv_results_dir}")

    # Use the full datasets as the base for LOOCV
    df_loocv_base_raw = df_raw_full.dropna(subset=['experiment_id']).copy() # Base raw data
    df_loocv_base_intact_fit = df_model_base_intact.dropna(subset=['experiment_id']).copy() # Base intact fit data

    experiment_ids_loocv = sorted(df_loocv_base_raw['experiment_id'].unique())
    n_folds = len(experiment_ids_loocv)
    print(f"Starting LOOCV with {n_folds} folds (experiments).")

    all_loocv_predictions = [] # List to store prediction DataFrames from each fold
    fold_metrics_intact = defaultdict(list) # Store metrics per fold
    fold_metrics_dna = defaultdict(list)
    fold_params = defaultdict(list) # Store fitted params per fold

    if n_folds < 2:
        print("LOOCV requires at least 2 experiments. Skipping.")
    else:
        for i, held_out_id in enumerate(experiment_ids_loocv):
            print(f"\n--- LOOCV Fold {i+1}/{n_folds}: Holding out Experiment {held_out_id} ---")

            # --- 1. Split Data ---
            # Training data: Exclude the held-out experiment
            train_ids = [eid for eid in experiment_ids_loocv if eid != held_out_id]
            df_intact_fit_train = df_loocv_base_intact_fit[df_loocv_base_intact_fit['experiment_id'].isin(train_ids)].copy()
            df_raw_train = df_loocv_base_raw[df_loocv_base_raw['experiment_id'].isin(train_ids)].copy()

            # Test data: Only the held-out experiment
            df_raw_test = df_loocv_base_raw[df_loocv_base_raw['experiment_id'] == held_out_id].copy()
            # df_intact_fit_test = df_loocv_base_intact_fit[df_loocv_base_intact_fit['experiment_id'] == held_out_id].copy() # We evaluate on df_raw_test

            if df_intact_fit_train.empty or df_raw_train.empty or df_raw_test.empty:
                print(f"  Skipping fold {i+1} due to empty train or test set after splitting.")
                continue

            # --- 2. Fit Models on Training Data ---
            print(f"  Fitting models on {len(df_intact_fit_train)} intact points and {len(df_raw_train)} raw points...")
            fit_results_fold = fit_models(
                df_intact_fit_train, df_raw_train,
                INITIAL_GUESS_INTACT, OPTIMIZATION_METHOD_INTACT,
                INITIAL_GUESS_C_RELEASE, INITIAL_GUESS_W_WASH, OPTIMIZATION_METHOD_DNA, BOUNDS_W_WASH
            )

            # Store fitted parameters for this fold
            fold_params['k'].append(fit_results_fold["k"])
            fold_params['alpha'].append(fit_results_fold["alpha"])
            fold_params['C_fresh'].append(fit_results_fold["C_release_fresh"])
            fold_params['C_frozen'].append(fit_results_fold["C_release_frozen"])
            fold_params['W_wash'].append(fit_results_fold["W_wash"])
            fold_params['success_intact'].append(fit_results_fold["success_intact"])
            fold_params['success_dna'].append(fit_results_fold["success_dna"])


            # --- 3. Predict on Held-Out (Test) Data ---
            print(f"  Predicting on {len(df_raw_test)} points from experiment {held_out_id}...")
            if not fit_results_fold["success_intact"]:
                print("   Skipping prediction and evaluation for this fold (intact fit failed).")
                # Append NaNs or empty results? Let's append NaNs to metrics lists
                metrics_intact_fold = calculate_metrics(pd.DataFrame()) # Will return NaNs
                metrics_dna_fold = calculate_dna_metrics(pd.DataFrame())
                df_pred_test = df_raw_test.copy() # Keep raw data structure
                df_pred_test["intact_frac_pred"] = np.nan
                df_pred_test["dna_pred"] = np.nan
            else:
                df_pred_test = predict_on_data(df_raw_test, fit_results_fold)
                 # --- 4. Evaluate on Test Data ---
                print("  Evaluating predictions...")
                metrics_intact_fold = calculate_metrics(df_pred_test) # Use the predictions on test data
                metrics_dna_fold = calculate_dna_metrics(df_pred_test) # Use the predictions on test data


            # --- 5. Store Fold Results ---
            all_loocv_predictions.append(df_pred_test) # Store the predictions for the held-out set

            print("  Fold Intact Metrics:", metrics_intact_fold)
            print("  Fold DNA Metrics:", metrics_dna_fold)

            # Append metrics to the overall lists
            for key, value in metrics_intact_fold.items():
                fold_metrics_intact[key].append(value)
            for key, value in metrics_dna_fold.items():
                fold_metrics_dna[key].append(value)

        # --- Post-LOOCV Aggregation and Reporting ---
        print("\n" + "="*80)
        print("--- LOOCV Summary ---")
        print("="*80)

        if not all_loocv_predictions:
            print("No LOOCV folds completed successfully.")
        else:
            # Combine all predictions into one DataFrame
            df_loocv_all_preds = pd.concat(all_loocv_predictions, ignore_index=True)

            # Save the combined predictions
            loocv_preds_path = loocv_results_dir / "loocv_predictions.csv"
            try:
                df_loocv_all_preds.to_csv(loocv_preds_path, index=False)
                print(f"Saved combined LOOCV predictions to {loocv_preds_path}")
            except Exception as e:
                 print(f"Error saving LOOCV predictions: {e}")

            # Calculate overall metrics based on the combined LOOCV predictions
            print("\nOverall LOOCV Metrics (Calculated across all folds' predictions):")
            overall_metrics_intact = calculate_metrics(df_loocv_all_preds)
            overall_metrics_dna = calculate_dna_metrics(df_loocv_all_preds)
            print("  Overall Intact:", overall_metrics_intact)
            print("  Overall DNA:", overall_metrics_dna)

            # Calculate average and std dev of metrics across folds
            print("\nAverage Metrics Across Folds:")
            avg_metrics_intact = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}" for k, v in fold_metrics_intact.items() if v}
            avg_metrics_dna = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}" for k, v in fold_metrics_dna.items() if v} # Adjust formatting if needed
            print("  Avg Fold Intact:", avg_metrics_intact)
            print("  Avg Fold DNA:", avg_metrics_dna)

            # Summarize fitted parameters across folds
            print("\nParameter Summary Across Folds (Mean ± Std Dev):")
            param_summary = {}
            for param, values in fold_params.items():
                 if param.startswith("success"): # Report count of successes
                      param_summary[param] = f"{sum(1 for v in values if v)} / {len(values)} successful"
                 else: # Report mean/std for numerical params
                      valid_values = [v for v in values if pd.notna(v)]
                      if valid_values:
                           mean_val = np.mean(valid_values)
                           std_val = np.std(valid_values)
                           # Adjust formatting based on parameter scale
                           if param in ['k']: fmt = "{:.3e} ± {:.3e}"
                           elif param in ['C_fresh', 'C_frozen']: fmt = "{:.1f} ± {:.1f}"
                           else: fmt = "{:.3f} ± {:.3f}"
                           param_summary[param] = fmt.format(mean_val, std_val)
                      else:
                           param_summary[param] = "N/A (No successful fits)"
            for p, v in param_summary.items(): print(f"  {p}: {v}")


            # Save LOOCV summary metrics
            loocv_summary = {
                "overall_metrics_intact": {k: float(v) if isinstance(v, (np.number, np.bool_)) else v for k,v in overall_metrics_intact.items()},
                "overall_metrics_dna": {k: float(v) if isinstance(v, (np.number, np.bool_)) else v for k,v in overall_metrics_dna.items()},
                "average_fold_metrics_intact": avg_metrics_intact,
                "average_fold_metrics_dna": avg_metrics_dna,
                "parameter_summary": param_summary,
                "n_folds": n_folds
            }
            loocv_summary_path = loocv_results_dir / "loocv_summary.json"
            try:
                with open(loocv_summary_path, 'w') as f:
                     json.dump(loocv_summary, f, indent=4, sort_keys=True)
                print(f"Saved LOOCV summary metrics to {loocv_summary_path}")
            except Exception as e: print(f"Error saving LOOCV summary: {e}")


            # --- Generate LOOCV Plots ---
            print("\nGenerating LOOCV Plots...")
            try:
                 # Use df_loocv_all_preds which has predictions generated during LOOCV
                 # Generate parity plots based on LOOCV predictions
                 plot_intact_parity(df_loocv_all_preds, loocv_results_dir)
                 plot_dna_parity_matplotlib(df_loocv_all_preds, loocv_results_dir, "LOOCV All Data")

                 # Maybe plot observed vs LOOCV predicted for each experiment? (Could be many plots)
                 # For now, just parity plots are sufficient.

            except Exception as e:
                 print(f"  ERROR during LOOCV plotting: {e}")
                 print("-" * 60); traceback.print_exc(); print("-" * 60)


    # --- Final Summary Table (Standard Fits) ---
    print("\n" + "="*80)
    print("--- Overall Fit Summary (Standard Fits on Subsets) ---")
    print("="*80)
    if not all_fit_results:
        print("No standard analysis configurations completed successfully.")
    else:
        summary_data = []
        for name, data in all_fit_results.items():
            metrics_intact = data.get('metrics_intact', {}) or {} # Handle None case
            metrics_dna = data.get('metrics_dna', {}) or {} # Handle None case
            summary_data.append({
                'Fit Config': name,
                'k': data.get('k', np.nan), 'alpha': data.get('alpha', np.nan),
                'C_fresh': data.get('C_release_fresh', np.nan), 'C_frozen': data.get('C_release_frozen', np.nan),
                'W_wash': data.get('W_wash', np.nan),
                'R² (Intact)': metrics_intact.get('R²', np.nan), 'RMSE (Intact)': metrics_intact.get('RMSE', np.nan), 'N (Intact)': metrics_intact.get('N_valid', 0),
                'R² (DNA)': metrics_dna.get('R²', np.nan), 'RMSE (DNA)': metrics_dna.get('RMSE', np.nan), 'MAE (DNA)': metrics_dna.get('MAE', np.nan), # Added MAE
                'RMSLE (DNA)': metrics_dna.get('RMSLE', np.nan), 'N (DNA)': metrics_dna.get('N_valid', 0), # Added RMSLE
                'Success': data.get('fit_successful_intact', False) and data.get('fit_successful_dna', False)
            })
        summary_df = pd.DataFrame(summary_data)

        formatters = {
            'k': '{:.3e}'.format, 'alpha': '{:.4f}'.format,
            'C_fresh': '{:.1f}'.format, 'C_frozen': '{:.1f}'.format, 'W_wash': '{:.3f}'.format,
            'R² (Intact)': '{:.4f}'.format, 'RMSE (Intact)': '{:.3f}'.format,
            'R² (DNA)': '{:.4f}'.format, 'RMSE (DNA)': '{:.1f}'.format, 'MAE (DNA)': '{:.1f}'.format, 'RMSLE (DNA)': '{:.4f}'.format,
        }
        for col, fmt in formatters.items():
            if col in summary_df.columns:
                 summary_df[col] = summary_df[col].apply(lambda x: fmt(x) if pd.notna(x) else 'NaN')
        print(summary_df.to_string(index=False, na_rep='NaN'))
    print("="*80)


    overall_end_time = time.time()
    print(f"\n--- Full Analysis Workflow Completed ({overall_end_time - overall_start_time:.2f} seconds) ---")
    print(f"Standard results saved in subdirectories within: {RESULTS_DIR}")
    print(f"LOOCV results saved in: {loocv_results_dir}")