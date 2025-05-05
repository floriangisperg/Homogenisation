# src/analysis/data_processing.py
import pandas as pd
import numpy as np
from pathlib import Path

# --- Define standard column names ---
# Add DNA columns to the map
COLUMN_MAP = {
    "Process step": "process_step",
    "total number of passages at 650 bar": "total_passages_650",
    "total number of passages at 1000 bar": "total_passages_1000",
    "Intact biomass percentage [%]": "intact_biomass_percent",
    "DNA [ng/µL]": "dna_conc", # Added DNA concentration
    "DNA std. dev. [ng/µL]": "dna_std_dev", # Added DNA std dev
    "wash procedure": "wash_procedure",
    "biomass type": "biomass_type",
    "experiment id": "experiment_id"
}

# Columns expected to be numeric after loading
NUMERIC_COLS = [
    "total_passages_650", "total_passages_1000",
    "intact_biomass_percent", "dna_conc", "dna_std_dev"
]
# Columns expected to be strings
STRING_COLS = ["process_step", "wash_procedure", "biomass_type"]

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads lysis data from an Excel file, renames columns,
    handles data types, and adds 'observed_frac'.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"  Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

    # Select and rename relevant columns based on the map keys that exist in the file
    mapped_cols_in_file = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing_expected_cols = set(COLUMN_MAP.keys()) - set(df.columns)
    if missing_expected_cols:
        print(f"Warning: The following expected columns were missing in the Excel file: {missing_expected_cols}")

    if not mapped_cols_in_file:
        raise ValueError("No mappable columns found in the Excel file based on COLUMN_MAP.")

    df = df[list(mapped_cols_in_file.keys())]
    df = df.rename(columns=mapped_cols_in_file)
    print(f"  Renamed columns to: {list(df.columns)}")

    # Convert numeric columns, coercing errors to NaN
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"  Warning: Found non-numeric values in column '{col}'; converted to NaN.")
        # else: # Removed warning spam if numeric col is missing - already warned above
        #     print(f"  Warning: Expected numeric column '{col}' not found.")

    # Clean string columns
    for col in STRING_COLS:
        if col in df.columns:
            # Fill potential NaN with empty string before stripping
            df[col] = df[col].fillna('').astype(str).str.strip()
            # Convert specific known strings to lowercase for consistency
            if col in ['wash_procedure', 'biomass_type']:
                 df[col] = df[col].str.lower()
        # else: # Removed warning spam
        #     print(f"  Warning: Expected string column '{col}' not found.")

    # Standardize specific biomass type strings if needed (already lowercase)
    # Example: df['biomass_type'] = df['biomass_type'].replace({'frozen':'frozen biomass'})

    # Calculate observed intact fraction
    if 'intact_biomass_percent' in df.columns:
        df["observed_frac"] = df["intact_biomass_percent"] / 100.0
    else:
         # Don't raise error immediately, maybe only fitting DNA
         print("Warning: Column 'intact_biomass_percent' needed for observed_frac not found.")
         df["observed_frac"] = np.nan # Create column with NaNs

    # Ensure experiment_id is suitable type (nullable integer)
    if 'experiment_id' in df.columns:
        # Try converting, handle potential errors if values are not convertible
        try:
            df["experiment_id"] = pd.to_numeric(df["experiment_id"], errors='coerce').astype('Int64')
            if df["experiment_id"].isnull().any():
                 print("Warning: Found non-numeric or missing values in 'experiment_id'.")
        except Exception as e:
            print(f"Warning: Could not convert 'experiment_id' to Int64: {e}")
    else:
         raise ValueError("Required column 'experiment_id' is missing.")


    print(f"  Data loading and basic processing complete. Shape: {df.shape}")
    # print("  Data types:\n", df.dtypes)
    # print("  Sample data:\n", df.head())
    return df

# --- Keep Intact Fraction Filtering Function ---
def filter_df_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Used ONLY for fitting intact fraction parameters k, alpha)
    For each experiment, if wash procedure is 'linear wash', keep only the
    first two rows (assuming chronological order based on index).
    For 'recursive wash', keep all rows.
    """
    if 'wash_procedure' not in df.columns or 'experiment_id' not in df.columns:
        raise ValueError("DataFrame must contain 'wash_procedure' and 'experiment_id' columns for filtering.")

    def filter_func(sub_df):
        # Sort by index within the group FIRST to ensure order
        sub_df_sorted = sub_df.sort_index()
        if sub_df_sorted.empty:
            return sub_df_sorted # Return empty if group is empty

        # Check the wash procedure of the first row (after sorting)
        wash_type = sub_df_sorted["wash_procedure"].iloc[0] # Already lowercase from load_data
        if wash_type == "linear wash":
            return sub_df_sorted.iloc[:2] # Return first two rows
        else:
            return sub_df_sorted # Return all rows for recursive or other types

    # Apply the function. group_keys=False avoids adding experiment_id to index.
    # Use observed=True if using pandas >= 1.5 to silence future warnings
    filtered_df = df.groupby("experiment_id", group_keys=False).apply(filter_func)#, observed=True)
    # Optional: Add a check here to be sure
    if 'experiment_id' not in filtered_df.columns:
        print("CRITICAL WARNING: 'experiment_id' column was dropped during filter_df_for_modeling!")
        # Potentially try merging it back if index is preserved? Or raise error.
        # For now, just warn. If this prints, the issue is deeper.

    return filtered_df

# --- Keep Dose Calculation Function ---
def add_cumulative_dose(df: pd.DataFrame, k: float, alpha: float) -> pd.DataFrame:
    """
    Calculates the cumulative dose for each row within an experiment.
    Assumes df is sorted by experiment_id and then temporally (by index).
    Adds a 'cumulative_dose' column. Uses standardized column names.
    """
    df_out = df.copy()
    df_out['cumulative_dose'] = 0.0 # Initialize column

    col_650 = "total_passages_650"
    col_1000 = "total_passages_1000"
    if col_650 not in df.columns or col_1000 not in df.columns:
         raise ValueError(f"DataFrame must contain '{col_650}' and '{col_1000}' columns.")

    all_doses = {} # Store doses per index

    for exp_id, group in df_out.groupby("experiment_id"):
        group_sorted = group.sort_index() # Ensure temporal order
        total_dose = 0.0
        prev_650 = 0
        prev_1000 = 0

        for idx, row in group_sorted.iterrows():
            inc_650 = row[col_650] - prev_650
            inc_1000 = row[col_1000] - prev_1000

            # Handle potential negative increments or NaNs
            inc_650 = max(0, inc_650) if pd.notna(inc_650) else 0
            inc_1000 = max(0, inc_1000) if pd.notna(inc_1000) else 0

            dose_inc = k * (650.0 ** alpha) * inc_650 + k * (1000.0 ** alpha) * inc_1000
            total_dose += dose_inc
            all_doses[idx] = total_dose # Map index to its cumulative dose

            prev_650 = row[col_650]
            prev_1000 = row[col_1000]

    # Assign calculated doses back to the original DataFrame using the index map
    df_out['cumulative_dose'] = df_out.index.map(all_doses)

    return df_out

# --- Add NEW function to calculate Delta F ---
def calculate_delta_F(df_with_F_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates F_before and delta_F = F_before - F_pred for DNA modeling.
    Requires df_with_F_pred to have columns: 'experiment_id', 'intact_frac_pred',
    'biomass_type', 'process_step', 'intact_biomass_percent', and index sorted chronologically.

    Adds columns: 'F_before', 'delta_F'.
    """
    if not all(col in df_with_F_pred.columns for col in ['experiment_id', 'intact_frac_pred', 'biomass_type', 'process_step', 'intact_biomass_percent']):
         raise ValueError("Missing required columns for calculate_delta_F")

    df_dna = df_with_F_pred.copy()
    df_dna["F_before"] = np.nan
    df_dna["delta_F"] = np.nan

    f0_col_name = "intact_biomass_percent" # Standardized name

    all_F_before = {} # Store results by index
    all_delta_F = {}

    for exp_id, group in df_dna.groupby("experiment_id"):
        # Sort the group by its original index HERE to ensure correct shift
        group_sorted = group.sort_index()
        if group_sorted.empty: continue

        # Get predicted F values from the *sorted* group, shift by 1 to get previous step's prediction
        F_pred_shifted = group_sorted["intact_frac_pred"].shift(1)

        # Determine F0 for this specific experiment
        first_row = group_sorted.iloc[0]
        if first_row["biomass_type"] == "fresh biomass": # Already lowercase
            F0 = 1.0
        else: # frozen biomass
            F0_observed = first_row[f0_col_name] / 100.0
            F0 = F0_observed if pd.notna(F0_observed) else np.nan # Use NaN if initial % is missing

        # --- Calculate F_before for each step in the sorted group ---
        f_before_values_group = {}
        for idx, row in group_sorted.iterrows():
            step_name = row["process_step"].strip().lower()
            f_before = np.nan # Default to NaN

            if step_name == "resuspended biomass":
                 f_before = F0 # F_before resuspended is the initial state F0
            else:
                 # F_before is the F_pred from the previous step (using shifted series)
                 f_before = F_pred_shifted.get(idx, np.nan) # Use .get for safety

            f_before_values_group[idx] = f_before

        # --- Calculate delta_F for each step in the sorted group ---
        delta_F_values_group = {}
        for idx, row in group_sorted.iterrows():
             f_before = f_before_values_group.get(idx, np.nan)
             f_pred = row["intact_frac_pred"] # Current predicted F
             delta_f = np.nan

             if pd.notna(f_before) and pd.notna(f_pred):
                 delta_f = f_before - f_pred
             # Override: delta_F for "resuspended biomass" step is considered 0 (no lysis *during* this step)
             if row["process_step"].strip().lower() == "resuspended biomass":
                 delta_f = 0.0

             delta_F_values_group[idx] = delta_f

        # Store group results in the main dictionaries
        all_F_before.update(f_before_values_group)
        all_delta_F.update(delta_F_values_group)

    # Assign calculated values back to the original DataFrame
    df_dna["F_before"] = df_dna.index.map(all_F_before)
    df_dna["delta_F"] = df_dna.index.map(all_delta_F)

    # Ensure delta_F is non-negative AFTER the loop
    df_dna["delta_F"] = df_dna["delta_F"].clip(lower=0)

    return df_dna

# --- Add helper to compute dose list (from previous step) ---
def compute_cumulative_dose_values(exp_data: pd.DataFrame, k: float, alpha: float) -> list[float]:
    """
    Computes a list of cumulative dose values for the rows in exp_data,
    ordered by the DataFrame's index. Uses standardized column names.
    """
    col_650 = "total_passages_650"
    col_1000 = "total_passages_1000"
    if col_650 not in exp_data.columns or col_1000 not in exp_data.columns:
        raise ValueError(f"Missing required columns for dose calculation: {col_650}, {col_1000}")

    cumulative_dose_list = []
    total_dose = 0.0
    prev_650 = 0
    prev_1000 = 0

    for idx, row in exp_data.sort_index().iterrows():
        inc_650 = row[col_650] - prev_650
        inc_1000 = row[col_1000] - prev_1000
        inc_650 = max(0, inc_650) if pd.notna(inc_650) else 0
        inc_1000 = max(0, inc_1000) if pd.notna(inc_1000) else 0
        dose_inc = k * (650.0 ** alpha) * inc_650 + k * (1000.0 ** alpha) * inc_1000
        total_dose += dose_inc
        cumulative_dose_list.append(total_dose)
        prev_650 = row[col_650]
        prev_1000 = row[col_1000]
    return cumulative_dose_list