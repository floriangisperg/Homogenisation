# src/analysis/models/intact_model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional, List, Union

from .base_model import Model
from ..data_processing import add_cumulative_dose


class IntactModel(Model):
    """
    Model for intact cell fraction prediction based on the mechanistic model:
    F = F₀ * exp(-k * P^α * N)

    where:
    - F₀ is the initial intact fraction (1.0 for fresh, <1.0 for frozen)
    - k is the lysis coefficient
    - α is the pressure exponent
    - P is the pressure
    - N is the number of passages
    """

    def __init__(self, name="intact_model"):
        super().__init__(name)
        self.params = {
            "k": np.nan,
            "alpha": np.nan
        }
        self.initial_guess = [1e-6, 1.5]  # [k, alpha]
        self.optimization_method = "Nelder-Mead"
        self.sse = np.nan

    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit the model parameters k and alpha to the given data.

        Args:
            data: DataFrame containing 'observed_frac', passage columns, etc.

        Returns:
            bool: True if fitting was successful, False otherwise
        """
        required_cols = ['experiment_id', 'observed_frac', 'total_passages_650',
                         'total_passages_1000', 'biomass_type']

        # DEBUG OUTPUT
        print("\n  DEBUG IntactModel.fit():")
        print(f"  Data shape: {data.shape}")
        print(f"  Data columns: {data.columns.tolist()}")

        # Check for presence of required columns
        if not all(col in data.columns for col in required_cols):
            missing = set(required_cols) - set(data.columns)
            print(f"Error: Missing required columns for IntactModel fitting: {missing}")
            return False

        if data.empty:
            print("Error: Empty DataFrame provided for IntactModel fitting.")
            return False

        # Check for NaN values in critical columns
        for col in required_cols:
            nan_count = data[col].isna().sum()
            print(f"  Column '{col}' has {nan_count} NaN values")
            if nan_count == len(data):
                print(f"Error: Column '{col}' contains only NaN values.")
                return False

        # Check for experiment_id values
        exp_ids = data['experiment_id'].unique()
        print(f"  Unique experiment_ids: {exp_ids}")

        # Check for valid numeric passages data
        valid_passages = (data['total_passages_650'].notna() | data['total_passages_1000'].notna()).any()
        if not valid_passages:
            print("Error: No valid passage data (both total_passages_650 and total_passages_1000 are all NaN).")
            return False

        self.training_data = data.copy()

        # Explicitly check observed_frac
        if data['observed_frac'].isna().all():
            print("Error: No valid observed_frac values (all NaN).")
            return False

        valid_data_count = (~data['observed_frac'].isna() &
                            (~data['total_passages_650'].isna() | ~data['total_passages_1000'].isna())).sum()
        print(f"  Valid data points for fitting: {valid_data_count}")

        if valid_data_count < 2:
            print("Error: Not enough valid data points for fitting (need at least 2).")
            return False

        try:
            # Optimize model parameters
            print("  Running optimization...")
            result = minimize(
                self._objective_function,
                x0=np.array(self.initial_guess),
                args=(data,),
                method=self.optimization_method,
                options={'disp': False, 'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7}
            )

            print(f"  Optimization result: success={result.success}, message={result.message}")

            if result.success:
                self.params["k"], self.params["alpha"] = result.x
                self.sse = result.fun
                self.fitted = True
                print(
                    f"IntactModel fitting successful: k={self.params['k']:.6e}, alpha={self.params['alpha']:.4f}, SSE={self.sse:.4f}")
                return True
            else:
                print(f"IntactModel fitting failed: {result.message}")
                return False

        except Exception as e:
            import traceback
            print(f"Error during IntactModel fitting: {e}")
            traceback.print_exc()
            return False

    def _objective_function(self, params: Tuple[float, float], df_filtered: pd.DataFrame) -> float:
        """
        Objective function for optimizing k and alpha parameters.

        Args:
            params: Tuple of (k, alpha)
            df_filtered: DataFrame with observed data

        Returns:
            float: Sum of squared errors between observations and predictions
        """
        k, alpha = params

        # Basic constraints
        if k <= 0 or alpha < 0:
            return np.inf

        # Verify data has required columns and is not empty
        if df_filtered.empty:
            print("ERROR: Empty dataframe in objective function!")
            return np.inf

        required_cols = ['observed_frac', 'total_passages_650', 'total_passages_1000']
        for col in required_cols:
            if col not in df_filtered.columns:
                print(f"ERROR: Missing required column '{col}' in objective function!")
                return np.inf

        try:
            # Check if we have enough valid data points
            valid_indices = ~df_filtered['observed_frac'].isna() & (
                        ~df_filtered['total_passages_650'].isna() | ~df_filtered['total_passages_1000'].isna())
            valid_count = valid_indices.sum()

            if valid_count < 2:
                print(f"ERROR: Not enough valid data points in objective function ({valid_count})!")
                return np.inf

            # Calculate cumulative dose using the current k, alpha
            # Explicitly pass only rows with valid data
            df_valid = df_filtered[valid_indices].copy()
            df_with_dose = add_cumulative_dose(df_valid, k, alpha)

            # Predict intact fraction
            df_pred = self._predict_intact_fraction(df_with_dose)

            # Calculate SSE
            valid_idx = df_pred["observed_frac"].notna() & df_pred["intact_frac_pred"].notna()
            if not valid_idx.any():
                print("ERROR: No valid prediction-observation pairs in objective function!")
                return np.inf

            observed = df_pred.loc[valid_idx, "observed_frac"]
            predicted = df_pred.loc[valid_idx, "intact_frac_pred"]
            residuals = observed - predicted
            sse = np.sum(residuals ** 2)

            # Print progress every 20 iterations
            if hasattr(self, '_iter_count'):
                self._iter_count += 1
                if self._iter_count % 20 == 0:
                    print(
                        f"  Iter {self._iter_count}, k={k:.3e}, alpha={alpha:.3f}, SSE={sse:.6f}, valid points={len(observed)}")
            else:
                self._iter_count = 1

            return sse if pd.notna(sse) else np.inf

        except Exception as e:
            print(f"Error in objective_function with k={k}, alpha={alpha}: {e}")
            import traceback
            traceback.print_exc()
            return np.inf

    def _predict_intact_fraction(self, df_with_dose: pd.DataFrame) -> pd.DataFrame:
        """
        Computes predicted intact fraction F = F₀ * exp(-cumulative_dose).

        Args:
            df_with_dose: DataFrame containing 'cumulative_dose', 'biomass_type' columns

        Returns:
            DataFrame with 'intact_frac_pred' column added
        """
        if 'cumulative_dose' not in df_with_dose.columns:
            raise ValueError("Input DataFrame must have 'cumulative_dose' column")

        df_pred = df_with_dose.copy()
        df_pred["intact_frac_pred"] = np.nan  # Initialize column

        for exp_id, group in df_pred.groupby("experiment_id"):
            group_sorted = group.sort_index()  # Ensure temporal order
            first_row = group_sorted.iloc[0]

            # Set initial intact fraction (F0) based on biomass type
            if first_row["biomass_type"] == "fresh biomass":
                F0 = 1.0
            elif first_row["biomass_type"] == "frozen biomass":
                # Use observed value from first step as F0
                F0 = first_row["intact_biomass_percent"] / 100.0 if "intact_biomass_percent" in first_row else np.nan
                if pd.isna(F0) or F0 <= 0 or F0 > 1:
                    F0 = first_row.get("observed_frac", np.nan)
                    if pd.isna(F0) or F0 <= 0 or F0 > 1:
                        print(f"Warning: Cannot determine valid F0 for frozen biomass in experiment {exp_id}")
                        continue
            else:
                print(f"Warning: Unknown biomass type '{first_row['biomass_type']}' in experiment {exp_id}")
                continue

            # Apply the formula using pre-calculated cumulative dose
            predicted_fractions = F0 * np.exp(-group_sorted["cumulative_dose"])

            # Assign predictions back to the original indices
            df_pred.loc[group_sorted.index, "intact_frac_pred"] = predicted_fractions

        return df_pred

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new data using the fitted model parameters.

        Args:
            data: DataFrame containing necessary columns for prediction

        Returns:
            DataFrame with 'intact_frac_pred' column added
        """
        if not self.has_required_params():
            print("Warning: Model parameters are not fully defined. Predictions may be invalid.")

        # Add cumulative dose using the fitted parameters
        df_with_dose = add_cumulative_dose(data, self.params["k"], self.params["alpha"])

        # Predict intact fraction
        return self._predict_intact_fraction(df_with_dose)

    def set_optimization_params(self, initial_guess=None, method=None):
        """
        Set optimization parameters for model fitting.

        Args:
            initial_guess: List of [k, alpha] initial values
            method: Optimization method to use
        """
        if initial_guess is not None:
            if len(initial_guess) != 2:
                raise ValueError("initial_guess must be a list of [k, alpha]")
            self.initial_guess = initial_guess

        if method is not None:
            self.optimization_method = method