# src/analysis/evaluation.py
import numpy as np
import pandas as pd
from typing import Dict

# --- Keep calculate_metrics for intact fraction ---
def calculate_metrics(df_pred: pd.DataFrame) -> Dict[str, float | int]:
    """
    Calculates SSE, R², RMSE, and MAE for intact fraction based on
    'observed_frac' and 'intact_frac_pred' columns. Handles NaNs.
    """
    obs_col = "observed_frac"
    pred_col = "intact_frac_pred"
    if obs_col not in df_pred.columns or pred_col not in df_pred.columns:
        print(f"Warning: Missing columns for intact metrics ({obs_col}, {pred_col}). Returning NaN.")
        return {"SSE": np.nan, "R²": np.nan, "RMSE": np.nan, "MAE": np.nan, "N_valid": 0}

    # Filter out rows where either observed or predicted is NaN
    valid_data = df_pred.dropna(subset=[obs_col, pred_col])

    if valid_data.empty:
        print("Warning: No valid data points found for intact metric calculation.")
        return {"SSE": np.nan, "R²": np.nan, "RMSE": np.nan, "MAE": np.nan, "N_valid": 0}

    observed = valid_data[obs_col]
    predicted = valid_data[pred_col]
    n_valid = len(observed)

    residuals = observed - predicted
    sse = np.sum(residuals**2)
    mean_observed = np.mean(observed)
    # Handle SST=0 case (e.g., all observed values are the same)
    sst = np.sum((observed - mean_observed)**2) if n_valid > 1 and np.var(observed) > 1e-12 else 0
    r2 = 1 - (sse / sst) if sst > 1e-12 else np.nan # Avoid division by zero or invalid R2
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    return {"SSE": sse, "R²": r2, "RMSE": rmse, "MAE": mae, "N_valid": n_valid}

# --- Add NEW function for DNA metrics ---
def calculate_dna_metrics(df_dna_pred: pd.DataFrame) -> Dict[str, float | int]:
    """
    Calculates error metrics for DNA prediction based on 'dna_conc' and 'dna_pred'.
    Excludes 'resuspended biomass' step. Handles NaNs. Adds RMSLE, MAPE, Bias.
    """
    obs_col = "dna_conc"
    pred_col = "dna_pred"
    step_col = "process_step"

    if obs_col not in df_dna_pred.columns or pred_col not in df_dna_pred.columns or step_col not in df_dna_pred.columns:
        print(f"Warning: Missing columns for DNA metrics ({obs_col}, {pred_col}, {step_col}). Returning NaN.")
        # Add new metrics as NaN too
        return {"SSE": np.nan, "R²": np.nan, "RMSE": np.nan, "MAE": np.nan,
                "RMSLE": np.nan, "MAPE": np.nan, "Bias": np.nan, "N_valid": 0}

    # Filter out NaNs and the resuspended step
    df_metrics = df_dna_pred.dropna(subset=[obs_col, pred_col]).copy()
    df_metrics = df_metrics[df_metrics[step_col].str.lower() != "resuspended biomass"]

    if df_metrics.empty:
        print("Warning: No valid data points found for DNA metric calculation after filtering.")
        return {"SSE": np.nan, "R²": np.nan, "RMSE": np.nan, "MAE": np.nan,
                "RMSLE": np.nan, "MAPE": np.nan, "Bias": np.nan, "N_valid": 0}

    observed = df_metrics[obs_col]
    predicted = df_metrics[pred_col]
    n_valid = len(observed)

    # --- Standard Metrics ---
    residuals = observed - predicted
    sse = np.sum(residuals**2)
    mean_observed = np.mean(observed)
    # Handle SST=0 case
    sst = np.sum((observed - mean_observed)**2) if n_valid > 1 and np.var(observed) > 1e-12 else 0
    r2 = 1 - (sse / sst) if sst > 1e-12 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals) # Mean Prediction Error (Obs - Pred)

    # --- Log/Percentage Metrics (handle potential issues) ---
    # RMSLE: Add 1 to avoid log(0)
    log_obs_p1 = np.log1p(observed) # log1p(x) = log(1+x)
    log_pred_p1 = np.log1p(predicted.clip(lower=0)) # Clip predicted to be non-negative before log1p
    rmsle = np.sqrt(np.mean((log_pred_p1 - log_obs_p1)**2))

    # MAPE: Avoid division by zero. Filter out observed values <= threshold
    mape_threshold = 1e-6 # Or some other small number relevant to your data scale
    valid_mape_idx = observed > mape_threshold
    if valid_mape_idx.any():
        mape = np.mean(np.abs((observed[valid_mape_idx] - predicted[valid_mape_idx]) / observed[valid_mape_idx])) * 100.0
    else:
        mape = np.nan # Or indicate insufficient data

    return {
        "SSE": sse, "R²": r2, "RMSE": rmse, "MAE": mae,
        "RMSLE": rmsle, "MAPE": mape, "Bias": bias,
        "N_valid": n_valid
    }