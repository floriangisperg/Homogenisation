# src/analysis/models/dna_model.py
import numpy as np
import pandas as pd

def predict_dna(df_dna_calc: pd.DataFrame, C_release_fresh: float, C_release_frozen: float, W_wash: float) -> pd.DataFrame:
    """
    Predicts DNA concentration based on C_release and W. Requires delta_F, F_before.
    Uses standardized column names.

    Args:
        df_dna_calc: DataFrame containing calculated 'delta_F', 'F_before',
                     'experiment_id', 'biomass_type', 'wash_procedure', 'process_step'.
        C_release_fresh: Fitted parameter for fresh biomass.
        C_release_frozen: Fitted parameter for frozen biomass.
        W_wash: Fitted wash efficiency factor.

    Returns:
        DataFrame with added 'dna_pred' column.
    """
    if not all(col in df_dna_calc.columns for col in ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step', 'delta_F', 'F_before']):
         raise ValueError("Missing required columns for predict_dna")

    df_pred = df_dna_calc.copy()
    df_pred["dna_pred"] = np.nan # Initialize prediction column

    all_dna_preds = {} # Store predictions by index

    for exp_id, group in df_pred.groupby("experiment_id"):
        group_sorted = group.sort_index() # Ensure chronological order
        if group_sorted.empty: continue

        dna_pred_prev = np.nan # Track prediction from the previous step for linear wash
        is_linear = group_sorted["wash_procedure"].iloc[0] == "linear wash" # Assumes consistent within exp
        is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"
        C_release = C_release_frozen if is_frozen else C_release_fresh

        # Determine F0 for this experiment (needed for resuspended step if frozen)
        # F_before for the first step *should* be F0
        first_row_idx = group_sorted.index[0]
        F0 = group_sorted.loc[first_row_idx, 'F_before'] # Get F0 from F_before of first step

        group_preds = {} # Store predictions for this group

        for idx, row in group_sorted.iterrows():
            current_pred = np.nan # Default to NaN
            step_name = row["process_step"].strip().lower() # Already lowercase
            delta_F_current = row["delta_F"] # Lysis fraction during this step

            # --- Determine prediction logic based on step and wash type ---
            if step_name == "resuspended biomass":
                if is_frozen:
                    # Model initial DNA based on freeze-thaw lysis (1 - F0)
                    if pd.notna(F0) and pd.notna(C_release_frozen):
                        # delta_F for freeze-thaw is effectively (1 - F0)
                        freeze_thaw_lysis_fraction = 1.0 - F0
                        freeze_thaw_lysis_fraction = max(0, freeze_thaw_lysis_fraction) # Ensure non-negative
                        current_pred = C_release_frozen * freeze_thaw_lysis_fraction
                    # else: keep current_pred = np.nan
                else: # Fresh biomass
                    # Assume F0=1, so freeze-thaw lysis = 0. Initial DNA = 0.
                    current_pred = 0.0

            # --- Predictions for HPH/Wash steps ---
            elif is_linear: # Linear Wash Procedure
                if step_name == "initial lysis":
                    # DNA released is proportional to lysis during this HPH step
                    if pd.notna(delta_F_current) and pd.notna(C_release):
                        current_pred = C_release * delta_F_current
                    # else: keep current_pred = np.nan
                else: # Subsequent linear wash steps (1st wash, 2nd wash, etc.)
                    # DNA concentration decays based on previous step's concentration and W
                    if pd.notna(dna_pred_prev) and pd.notna(W_wash):
                        current_pred = dna_pred_prev * W_wash
                    # else: keep current_pred = np.nan
            else: # Recursive Wash Procedure (and potentially other types)
                 # DNA released is proportional to lysis during this HPH step
                 if pd.notna(delta_F_current) and pd.notna(C_release):
                     current_pred = C_release * delta_F_current
                 # else: keep current_pred = np.nan

            # --- Store prediction and update previous prediction ---
            group_preds[idx] = current_pred
            dna_pred_prev = current_pred # Update for the next step in linear wash

        # Update the main dictionary
        all_dna_preds.update(group_preds)

    # Assign predicted values back to the original DataFrame
    df_pred["dna_pred"] = df_pred.index.map(all_dna_preds)

    return df_pred