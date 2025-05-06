# src/analysis/models/dna_model_step_dep.py
import numpy as np
import pandas as pd

def predict_dna_step_dep(df_dna_calc: pd.DataFrame, C_release_fresh: float, C_release_frozen: float,
                         W_wash_1st: float, W_wash_subsequent: float) -> pd.DataFrame:
    """
    Predicts DNA concentration using STEP-DEPENDENT wash efficiency (W_1st, W_subsequent).
    Requires delta_F, F_before. Uses standardized column names.

    Args:
        df_dna_calc: DataFrame containing input columns.
        C_release_fresh: Fitted parameter for fresh biomass.
        C_release_frozen: Fitted parameter for frozen biomass.
        W_wash_1st: Fitted wash efficiency factor for the FIRST linear wash step.
        W_wash_subsequent: Fitted wash efficiency factor for SUBSEQUENT linear wash steps.

    Returns:
        DataFrame with added 'dna_pred' column.
    """
    if not all(col in df_dna_calc.columns for col in ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step', 'delta_F', 'F_before']):
         raise ValueError("Missing required columns for predict_dna_step_dep")

    df_pred = df_dna_calc.copy()
    df_pred["dna_pred"] = np.nan

    all_dna_preds = {}

    for exp_id, group in df_pred.groupby("experiment_id"):
        group_sorted = group.sort_index()
        if group_sorted.empty: continue

        dna_pred_prev = np.nan
        is_linear = group_sorted["wash_procedure"].iloc[0] == "linear wash"
        is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"
        C_release = C_release_frozen if is_frozen else C_release_fresh

        first_row_idx = group_sorted.index[0]
        F0 = group_sorted.loc[first_row_idx, 'F_before']

        group_preds = {}
        is_first_linear_wash_step_encountered = False

        for idx, row in group_sorted.iterrows():
            current_pred = np.nan
            step_name = row["process_step"].strip().lower()
            delta_F_current = row["delta_F"]

            if step_name == "resuspended biomass":
                if is_frozen:
                    if pd.notna(F0) and pd.notna(C_release_frozen):
                        freeze_thaw_lysis_fraction = max(0, 1.0 - F0)
                        current_pred = C_release_frozen * freeze_thaw_lysis_fraction
                else: current_pred = 0.0

            elif is_linear:
                if step_name == "initial lysis":
                    if pd.notna(delta_F_current) and pd.notna(C_release):
                        current_pred = C_release * delta_F_current
                    is_first_linear_wash_step_encountered = False
                else: # Linear wash steps
                    # Determine which W parameter to use
                    if not is_first_linear_wash_step_encountered:
                        W_current = W_wash_1st
                        is_first_linear_wash_step_encountered = True
                    else:
                        W_current = W_wash_subsequent

                    if pd.notna(dna_pred_prev) and pd.notna(W_current):
                        current_pred = dna_pred_prev * W_current

            else: # Recursive Wash Procedure
                 if pd.notna(delta_F_current) and pd.notna(C_release):
                     current_pred = C_release * delta_F_current

            group_preds[idx] = current_pred
            if pd.notna(current_pred):
                dna_pred_prev = current_pred

        all_dna_preds.update(group_preds)

    df_pred["dna_pred"] = df_pred.index.map(all_dna_preds)
    if "dna_pred" in df_pred.columns:
        df_pred["dna_pred"] = df_pred["dna_pred"].clip(lower=0)

    return df_pred