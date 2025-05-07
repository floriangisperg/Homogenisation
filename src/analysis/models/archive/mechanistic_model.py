# src/analysis/models/mechanistic_model.py
import numpy as np
import pandas as pd

def predict_intact_fraction(df: pd.DataFrame, k: float, alpha: float) -> pd.DataFrame:
    """
    Computes predicted intact fraction F = F₀ * exp(-cumulative_dose).
    Requires 'cumulative_dose' and 'biomass_type' columns,
    and 'intact_biomass_percent' for frozen biomass F0 calculation.
    Adds an 'intact_frac_pred' column.
    Assumes df is sorted by experiment_id and temporally (index).
    """
    if 'cumulative_dose' not in df.columns:
        raise ValueError("Input DataFrame must have a 'cumulative_dose' column. Run 'add_cumulative_dose' first.")
    if 'biomass_type' not in df.columns or 'intact_biomass_percent' not in df.columns:
         raise ValueError("Input DataFrame must have 'biomass_type' and 'intact_biomass_percent' columns.")

    df_pred = df.copy()
    df_pred["intact_frac_pred"] = np.nan # Initialize column

    for exp_id, group in df_pred.groupby("experiment_id"):
        group_sorted = group.sort_index() # Ensure temporal order
        first_row = group_sorted.iloc[0]

        if first_row["biomass_type"] == "fresh biomass":
            F0 = 1.0
        elif first_row["biomass_type"] == "frozen biomass":
            # Use the observed value from the first step as F0
            F0 = first_row["intact_biomass_percent"] / 100.0
            if pd.isna(F0) or F0 <= 0 or F0 > 1:
                print(f"Warning: Invalid F0={F0} calculated for frozen biomass in experiment {exp_id}. Using observed_frac: {first_row.get('observed_frac', 'N/A')}. Check first data point.")
                # Fallback or raise error depending on desired robustness
                F0 = first_row["observed_frac"] # Assumes observed_frac exists
                if pd.isna(F0) or F0 <= 0 or F0 > 1:
                     raise ValueError(f"Cannot determine valid F0 for frozen biomass in experiment {exp_id}")
        else:
             raise ValueError(f"Unknown biomass type '{first_row['biomass_type']}' in experiment {exp_id}")

        # Apply the formula using the pre-calculated cumulative dose
        predicted_fractions = F0 * np.exp(-group_sorted["cumulative_dose"])

        # Assign predictions back to the original indices in df_pred
        df_pred.loc[group_sorted.index, "intact_frac_pred"] = predicted_fractions

    return df_pred