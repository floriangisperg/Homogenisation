# src/analysis/models/simplified_compartment_model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional, List, Union

from .dna_models import DNABaseModel
from ..data_processing import calculate_delta_F


class SimplifiedCompartmentModel(DNABaseModel):
    """
    Simplified two-compartment DNA model with fewer parameters.
    This model maintains the conceptual framework but uses:
    - Single adsorption fraction per biomass type
    - Single desorption rate per biomass type
    - Simplified wash parameters: first wash vs. subsequent washes
    """

    def __init__(self, name="simplified_compartment_model"):
        super().__init__(name)

        # Add model parameters
        self.params.update({
            # Adsorption parameters
            "adsorption_fraction_fresh": np.nan,
            "adsorption_fraction_frozen": np.nan,

            # Wash efficiency parameters (simplified)
            "W_first_wash": np.nan,
            "W_subsequent_wash": np.nan,

            # Desorption parameters
            "D_fresh": np.nan,
            "D_frozen": np.nan,
        })

        # Initial guesses for optimization
        self.initial_guess = [
            # Adsorption fractions
            0.3, 0.5,  # adsorption_fraction_fresh, adsorption_fraction_frozen

            # Wash efficiencies
            0.6, 0.4,  # W_first_wash, W_subsequent_wash

            # Desorption rates
            0.2, 0.1  # D_fresh, D_frozen
        ]

        # Parameter bounds for optimization
        self.bounds = [
            # Adsorption fractions (0-1)
            (0.01, 0.99), (0.01, 0.99),

            # Wash efficiencies (0-1)
            (0.01, 0.99), (0.01, 0.99),

            # Desorption rates (0-1)
            (0.01, 0.99), (0.01, 0.99)
        ]

        self.optimization_method = "L-BFGS-B"

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit model parameters to the given data.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for SimplifiedCompartmentModel fitting: {missing}")
            return False

        try:
            self.training_data = data_with_intact_pred.copy()

            # Calculate delta_F from intact_frac_pred
            df_dna_base = calculate_delta_F(self.training_data)

            # 1. First fit C_release parameters (using base class method)
            success_C = self._fit_C_release_parameters(df_dna_base)
            if not success_C:
                print("Warning: Failed to fit C_release parameters")
                # We can still proceed with default values or previously set values

            # 2. Fit the compartment model parameters
            success_compartment = self._fit_compartment_parameters(df_dna_base)

            # Overall success based on compartment parameter fitting
            self.fitted = success_compartment
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during SimplifiedCompartmentModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_compartment_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the parameters for the simplified compartment model.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        print("Fitting simplified compartment model parameters...")

        # Prepare data: we need all sequential steps within experiments
        if df_dna_base.empty:
            print("Error: Empty DataFrame for compartment parameter fitting")
            return False

        # Check that we have wash steps
        if not any(s.lower().endswith('wash') for s in df_dna_base['process_step'].unique()):
            print("Error: No wash steps found in data")
            return False

        try:
            # Only fit using linear wash data which has sequential steps
            df_fit_data = df_dna_base[df_dna_base['wash_procedure'] == 'linear wash'].copy()

            # Filter out resuspended biomass (handled separately)
            df_fit_data = df_fit_data[df_fit_data['process_step'].str.lower() != 'resuspended biomass'].copy()

            if df_fit_data.empty:
                print("Error: No valid data for compartment parameter fitting after filtering")
                return False

            # Run optimization
            result = minimize(
                self._objective_compartment_model,
                x0=np.array(self.initial_guess),
                args=(df_fit_data,),
                method=self.optimization_method,
                bounds=self.bounds
            )

            if result.success:
                # Unpack and assign parameters
                (
                    self.params["adsorption_fraction_fresh"],
                    self.params["adsorption_fraction_frozen"],

                    self.params["W_first_wash"],
                    self.params["W_subsequent_wash"],

                    self.params["D_fresh"],
                    self.params["D_frozen"]
                ) = result.x

                print("Simplified compartment model parameters fitted successfully")
                print(f"  Adsorption fractions - Fresh: {self.params['adsorption_fraction_fresh']:.3f}, "
                      f"Frozen: {self.params['adsorption_fraction_frozen']:.3f}")
                print(f"  Wash efficiencies - First: {self.params['W_first_wash']:.3f}, "
                      f"Subsequent: {self.params['W_subsequent_wash']:.3f}")
                print(
                    f"  Desorption rates - Fresh: {self.params['D_fresh']:.3f}, Frozen: {self.params['D_frozen']:.3f}")
                print(f"  Objective function value: {result.fun:.4f}")

                return True
            else:
                print(f"Error: Compartment parameter fitting failed: {result.message}")
                return False

        except Exception as e:
            print(f"Error during compartment parameter fitting: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _objective_compartment_model(self, params: List[float], df_dna_model: pd.DataFrame) -> float:
        """
        Objective function for fitting compartment model parameters.

        Args:
            params: List with parameter values
            df_dna_model: DataFrame with DNA data

        Returns:
            float: Sum of squared errors
        """
        # Unpack parameters
        (
            adsorption_fraction_fresh, adsorption_fraction_frozen,
            W_first_wash, W_subsequent_wash,
            D_fresh, D_frozen
        ) = params

        # Basic constraints
        if not all(0 < p < 1 for p in params):
            return np.inf

        # Create parameter dictionary for easy access
        param_dict = {
            "adsorption_fraction_fresh": adsorption_fraction_fresh,
            "adsorption_fraction_frozen": adsorption_fraction_frozen,
            "W_first_wash": W_first_wash,
            "W_subsequent_wash": W_subsequent_wash,
            "D_fresh": D_fresh,
            "D_frozen": D_frozen,
            "C_release_fresh": self.params["C_release_fresh"],
            "C_release_frozen": self.params["C_release_frozen"]
        }

        # Make predictions with current parameters
        df_pred = self._predict_with_params(df_dna_model, param_dict)

        # Calculate SSE
        valid_idx = df_pred["dna_conc"].notna() & df_pred["dna_pred"].notna()
        if not valid_idx.any():
            return np.inf

        observed = df_pred.loc[valid_idx, "dna_conc"]
        predicted = df_pred.loc[valid_idx, "dna_pred"]
        residuals = observed - predicted
        sse = np.sum(residuals ** 2)

        return sse

    def _predict_with_params(self, df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """
        Make predictions using the given parameters.

        Args:
            df: DataFrame with intact fraction data
            params: Dictionary of parameter values

        Returns:
            DataFrame with dna_pred column added
        """
        df_pred = df.copy()
        df_pred["dna_pred"] = np.nan

        # Track compartment values per experiment
        exp_compartments = {}

        for exp_id, group in df_pred.groupby("experiment_id"):
            group_sorted = group.sort_index()

            if group_sorted.empty:
                continue

            # Determine biomass type for this experiment
            is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"

            # Get biomass-specific parameters
            C_release = params["C_release_frozen"] if is_frozen else params["C_release_fresh"]
            adsorption_fraction = params["adsorption_fraction_frozen"] if is_frozen else params[
                "adsorption_fraction_fresh"]
            D = params["D_frozen"] if is_frozen else params["D_fresh"]

            # Initialize compartment values for this experiment
            DNA_free = 0.0
            DNA_bound = 0.0

            # Initialize wash step counter
            wash_step_number = 0

            for idx, row in group_sorted.iterrows():
                step_name = row["process_step"].strip().lower()
                delta_F = row["delta_F"]
                wash_procedure = row["wash_procedure"].strip().lower()

                if step_name == "resuspended biomass":
                    if is_frozen:
                        # For frozen biomass, account for initial release due to freeze-thaw
                        F0 = row["F_before"]
                        freeze_thaw_lysis = max(0, 1.0 - F0)
                        DNA_released = C_release * freeze_thaw_lysis

                        # Partition released DNA into free and bound compartments
                        DNA_free = DNA_released * (1 - adsorption_fraction)
                        DNA_bound = DNA_released * adsorption_fraction
                    else:
                        # For fresh biomass, assume no initial DNA release
                        DNA_free = 0.0
                        DNA_bound = 0.0

                    # Prediction for this step is just the free DNA
                    df_pred.loc[idx, "dna_pred"] = DNA_free

                elif step_name == "initial lysis":
                    # Calculate newly released DNA from lysis
                    if pd.notna(delta_F) and pd.notna(C_release):
                        DNA_released = C_release * delta_F

                        # Partition newly released DNA
                        DNA_free += DNA_released * (1 - adsorption_fraction)
                        DNA_bound += DNA_released * adsorption_fraction

                    # Reset wash step counter
                    wash_step_number = 0

                    # Prediction for this step is just the free DNA
                    df_pred.loc[idx, "dna_pred"] = DNA_free

                elif "wash" in step_name and wash_procedure == "linear wash":
                    # Increment wash step counter
                    wash_step_number += 1

                    # Get appropriate wash efficiency parameter
                    W = params["W_first_wash"] if wash_step_number == 1 else params["W_subsequent_wash"]

                    if pd.notna(W):
                        # Calculate washout of free DNA
                        DNA_washout = DNA_free * W
                        DNA_free -= DNA_washout

                        # Calculate desorption from bound DNA
                        DNA_desorption = DNA_bound * (1 - np.exp(-D * wash_step_number))
                        DNA_bound -= DNA_desorption
                        DNA_free += DNA_desorption

                    # Prediction is the current free DNA
                    df_pred.loc[idx, "dna_pred"] = DNA_free

                elif "wash" in step_name and wash_procedure == "recursive wash":
                    # For recursive wash, treat as combined lysis + wash
                    if pd.notna(delta_F) and pd.notna(C_release):
                        # New DNA release from lysis
                        DNA_released = C_release * delta_F

                        # Partition newly released DNA
                        DNA_free += DNA_released * (1 - adsorption_fraction)
                        DNA_bound += DNA_released * adsorption_fraction

                    # Prediction is the current free DNA
                    df_pred.loc[idx, "dna_pred"] = DNA_free

            # Store final compartment values for debugging
            exp_compartments[exp_id] = {
                "DNA_free": DNA_free,
                "DNA_bound": DNA_bound
            }

        return df_pred

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the simplified compartment model.

        Args:
            data_with_intact_pred: DataFrame with intact_frac_pred column

        Returns:
            DataFrame with dna_pred column added
        """
        if not self.has_required_params():
            print("Warning: Model parameters are not fully defined. Predictions may be invalid.")
            missing_params = [k for k, v in self.params.items() if pd.isna(v)]
            print(f"Missing parameters: {missing_params}")

        # Prepare data with delta_F
        df_dna_calc = calculate_delta_F(data_with_intact_pred)

        # Make predictions using current parameters
        return self._predict_with_params(df_dna_calc, self.params)