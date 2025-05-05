# src/analysis/optimization.py
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .data_processing import add_cumulative_dose # Make sure this is imported
from .models.mechanistic_model import predict_intact_fraction
import traceback # Optional: for more detailed error printing if needed

# --- Corrected objective_intact function ---
def objective_intact(params: tuple[float, float], df_filtered: pd.DataFrame) -> float:
    """
    Objective for Intact Fraction Model - uses the filtered df.
    Calculates cumulative dose internally based on current k, alpha params.
    Expects df_filtered to have 'observed_frac', 'biomass_type', passage columns.
    """
    k, alpha = params
    # Basic constraints (can be handled by optimizer bounds too)
    if k <= 0 or alpha < 0: # k must be positive, alpha typically non-negative
        return np.inf

    try:
        # 1. Calculate cumulative dose using the current k, alpha
        df_with_dose = add_cumulative_dose(df_filtered, k, alpha)

        # 2. Predict intact fraction using the dataframe that now HAS the dose
        df_pred = predict_intact_fraction(df_with_dose, k, alpha)

        # 3. Calculate SSE
        obs_col_name = "observed_frac" # Standardized name
        pred_col_name = "intact_frac_pred"

        if obs_col_name not in df_pred.columns or pred_col_name not in df_pred.columns:
            # This shouldn't happen if previous steps worked, but good sanity check
            print(f"Warning: Missing columns '{obs_col_name}' or '{pred_col_name}' in objective_intact.")
            return 1e12

        # Ensure we only compare where observed data exists AND prediction was possible
        valid_idx = df_pred[obs_col_name].notna() & df_pred[pred_col_name].notna()
        if not valid_idx.any():
            # print(f"Warning: No valid observed/predicted pairs in objective_intact for k={k:.2e}, alpha={alpha:.2f}")
            return 1e12 # Return large number if no valid points

        observed = df_pred.loc[valid_idx, obs_col_name]
        predicted = df_pred.loc[valid_idx, pred_col_name]
        residuals = observed - predicted
        sse = np.sum(residuals**2)

        # Optional: Add penalty for unrealistic parameters if needed
        # e.g., if alpha gets too large: if alpha > 10: sse *= (1 + (alpha - 10))

        # Return NaN if SSE calculation results in NaN (shouldn't usually happen here)
        return sse if pd.notna(sse) else np.inf


    except Exception as e:

        # Print more details if an unexpected error occurs during calculation

        print(f"Error in objective_intact with k={k}, alpha={alpha}: {e}")

        print("-" * 60)

        traceback.print_exc()  # <<< ADD THIS LINE TO PRINT THE FULL TRACEBACK

        print("-" * 60)

        return np.inf  # Return infinity on error to guide optimizer away

# --- Keep other functions (objective_dna_release, objective_dna_wash, fit_intact_model) ---
# Ensure fit_intact_model uses the corrected objective_intact function name

def objective_dna_release(params: list[float], df_dna_model: pd.DataFrame, biomass_type_filter: str) -> float:
    """
    Objective for C_release fitting. Uses standardized column names.
    Expects df_dna_model to have 'dna_conc', 'delta_F', 'biomass_type'.
    """
    C_release = params[0]
    if C_release < 0: return np.inf # Constraint

    df_subset = df_dna_model[df_dna_model["biomass_type"] == biomass_type_filter].copy()
    if df_subset.empty: return 1e12

    df_subset["dna_pred"] = C_release * df_subset["delta_F"]
    obs_col = "dna_conc"; pred_col = "dna_pred"
    valid_idx = df_subset[obs_col].notna() & df_subset[pred_col].notna()
    if not valid_idx.any(): return 1e12

    observed = df_subset.loc[valid_idx, obs_col]
    predicted = df_subset.loc[valid_idx, pred_col]
    residuals = observed - predicted
    return np.sum(residuals**2)

def objective_dna_wash(params: list[float], df_linear_wash_decay: pd.DataFrame) -> float:
    """
    Objective for wash efficiency factor W. Uses standardized column names.
    Expects df_linear_wash_decay with 'dna_conc', 'dna_pred_release' (DNA conc after initial lysis).
    """
    W = params[0]
    if not (0 <= W <= 1): return np.inf

    sse = 0.0
    total_comparisons = 0
    predicted_dna_values = {}

    for exp_id, group in df_linear_wash_decay.groupby("experiment_id"):
        group_sorted = group.sort_index()
        if group_sorted.empty or 'dna_pred_release' not in group_sorted.columns: continue

        dna_pred_prev = group_sorted['dna_pred_release'].iloc[0]
        if pd.isna(dna_pred_prev):
             # print(f"Warning: NaN seed value 'dna_pred_release' for Exp {exp_id} in objective_dna_wash. Skipping group.")
             continue

        for idx, row in group_sorted.iterrows():
            dna_pred_current = dna_pred_prev * W
            predicted_dna_values[idx] = dna_pred_current
            observed_dna = row["dna_conc"]
            if pd.notna(observed_dna):
                sse += (observed_dna - dna_pred_current)**2
                total_comparisons += 1
            dna_pred_prev = dna_pred_current

    return sse if total_comparisons > 0 else 1e12


def fit_intact_model(df_model: pd.DataFrame, initial_guess: list[float] = [1e-5, 2.0], method: str = "Nelder-Mead") -> tuple[float, float, float, bool]:
    """ Fits the model parameters k and alpha using the objective_intact function. """
    print(f"Starting optimization for k, alpha with initial guess: {initial_guess} using method '{method}'...")

    result = minimize(
        objective_intact, # Ensure this uses the corrected function above
        x0=np.array(initial_guess),
        args=(df_model,),
        method=method,
        options={'disp': False, 'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7}
    )

    if result.success:
        best_k, best_alpha = result.x
        final_sse = result.fun
        print(f"Optimization successful!")
        print(f"  Best k: {best_k:.6e}")
        print(f"  Best alpha: {best_alpha:.4f}")
        print(f"  Final SSE (Intact Fit): {final_sse:.4f}")
    else:
        print(f"Optimization failed for k, alpha: {result.message}")
        best_k, best_alpha, final_sse = np.nan, np.nan, np.nan

    return best_k, best_alpha, final_sse, result.success