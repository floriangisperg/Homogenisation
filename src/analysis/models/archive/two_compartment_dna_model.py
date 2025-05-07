# src/analysis/models/two_compartment_mechanistic_model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List

from analysis.models.dna_models import DNABaseModel
from analysis.data_processing import calculate_delta_F


class TwoCompartmentMechanisticModel(DNABaseModel):
    """
    Two-compartment mechanistic DNA model that accounts for:
    1. DNA release from cell lysis
    2. DNA partitioning between free and bound states
    3. Different wash efficiencies based on concentration
    4. Desorption of bound DNA affected by mechanical disruption
    5. Different behavior for linear vs. recursive washing

    This model explains why recursive processing may recover more total DNA
    than linear washing despite similar lysis extent.
    """

    def __init__(self, name="two_compartment_mechanistic_model"):
        super().__init__(name)

        # Add model parameters
        self.params.update({
            # DNA release parameters (from base model)
            # "C_release_fresh": np.nan,  # Already in base model
            # "C_release_frozen": np.nan, # Already in base model

            # Adsorption parameters
            "k_ads_fresh": np.nan,  # Adsorption coefficient for fresh biomass
            "k_ads_frozen": np.nan,  # Adsorption coefficient for frozen biomass

            # Desorption parameters
            "k_des_fresh": np.nan,  # Base desorption coefficient for fresh biomass
            "k_des_frozen": np.nan,  # Base desorption coefficient for frozen biomass

            # Mechanical disruption factors
            "M_factor_homogenization": np.nan,  # Mechanical factor during homogenization
            "M_factor_wash": np.nan,  # Mechanical factor during wash only

            # Wash efficiency parameters (concentration-dependent)
            "W_min": np.nan,  # Minimum wash efficiency (high concentration)
            "W_max": np.nan,  # Maximum wash efficiency (low concentration)
            "beta": np.nan,  # Concentration sensitivity parameter
        })

        # Initial guesses for optimization
        self.initial_guess = [
            # Adsorption coefficients (0-1)
            0.3, 0.5,  # k_ads_fresh, k_ads_frozen

            # Desorption coefficients (0-1)
            0.2, 0.1,  # k_des_fresh, k_des_frozen

            # Mechanical factors (relative values)
            1.0, 0.2,  # M_factor_homogenization, M_factor_wash

            # Wash efficiency parameters
            0.3, 0.8, 0.001  # W_min, W_max, beta
        ]

        # Parameter bounds for optimization
        self.bounds = [
            # Adsorption coefficients (0-1)
            (0.01, 0.99), (0.01, 0.99),

            # Desorption coefficients (0-1)
            (0.01, 0.99), (0.01, 0.99),

            # Mechanical factors (0+)
            (0.1, 5.0), (0.01, 1.0),

            # Wash efficiency parameters
            (0.01, 0.99), (0.01, 0.99), (0.0001, 0.1)
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
                         'wash_procedure', 'process_step', 'dna_conc',
                         'cumulative_dose']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for TwoCompartmentMechanisticModel fitting: {missing}")
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

            # 2. Fit the two-compartment model parameters
            success_compartment = self._fit_compartment_parameters(df_dna_base)

            # Overall success based on compartment parameter fitting
            self.fitted = success_compartment
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during TwoCompartmentMechanisticModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_compartment_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the parameters for the two-compartment model.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        print("Fitting two-compartment mechanistic model parameters...")

        # Prepare data: we need all sequential steps within experiments
        if df_dna_base.empty:
            print("Error: Empty DataFrame for compartment parameter fitting")
            return False

        # Check that we have wash steps
        if not any(s.lower().find('wash') >= 0 for s in df_dna_base['process_step'].unique()):
            print("Error: No wash steps found in data")
            return False

        try:
            # Run optimization
            result = minimize(
                self._objective_compartment_model,
                x0=np.array(self.initial_guess),
                args=(df_dna_base,),
                method=self.optimization_method,
                bounds=self.bounds
            )

            if result.success:
                # Unpack and assign parameters
                (
                    self.params["k_ads_fresh"],
                    self.params["k_ads_frozen"],

                    self.params["k_des_fresh"],
                    self.params["k_des_frozen"],

                    self.params["M_factor_homogenization"],
                    self.params["M_factor_wash"],

                    self.params["W_min"],
                    self.params["W_max"],
                    self.params["beta"]
                ) = result.x

                print("Two-compartment mechanistic model parameters fitted successfully")
                print(f"  Adsorption coefficients - Fresh: {self.params['k_ads_fresh']:.3f}, "
                      f"Frozen: {self.params['k_ads_frozen']:.3f}")
                print(f"  Desorption coefficients - Fresh: {self.params['k_des_fresh']:.3f}, "
                      f"Frozen: {self.params['k_des_frozen']:.3f}")
                print(f"  Mechanical factors - Homogenization: {self.params['M_factor_homogenization']:.3f}, "
                      f"Wash: {self.params['M_factor_wash']:.3f}")
                print(f"  Wash parameters - W_min: {self.params['W_min']:.3f}, "
                      f"W_max: {self.params['W_max']:.3f}, beta: {self.params['beta']:.6f}")
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
            k_ads_fresh, k_ads_frozen,
            k_des_fresh, k_des_frozen,
            M_factor_homogenization, M_factor_wash,
            W_min, W_max, beta
        ) = params

        # Basic constraints to ensure physically meaningful parameters
        if (not (0 < k_ads_fresh < 1) or not (0 < k_ads_frozen < 1) or
                not (0 < k_des_fresh < 1) or not (0 < k_des_frozen < 1) or
                not (M_factor_homogenization > 0) or not (M_factor_wash > 0) or
                not (0 < W_min < 1) or not (0 < W_max < 1) or not (beta > 0) or
                not (W_min < W_max)):  # Ensure W_min < W_max
            return np.inf

        # Create parameter dictionary for easier access
        param_dict = {
            "k_ads_fresh": k_ads_fresh,
            "k_ads_frozen": k_ads_frozen,
            "k_des_fresh": k_des_fresh,
            "k_des_frozen": k_des_frozen,
            "M_factor_homogenization": M_factor_homogenization,
            "M_factor_wash": M_factor_wash,
            "W_min": W_min,
            "W_max": W_max,
            "beta": beta,
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

    def _calculate_wash_efficiency(self, dna_conc: float, W_min: float, W_max: float, beta: float) -> float:
        """
        Calculate concentration-dependent wash efficiency.

        Args:
            dna_conc: Current DNA concentration
            W_min: Minimum wash efficiency (at high concentration)
            W_max: Maximum wash efficiency (at low concentration)
            beta: Concentration sensitivity parameter

        Returns:
            float: Wash efficiency value
        """
        if dna_conc <= 0 or pd.isna(dna_conc):
            return W_max  # Default to maximum efficiency for zero/invalid concentration

        W = W_min + (W_max - W_min) * np.exp(-beta * dna_conc)
        return W

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
            group_sorted = group.sort_index()  # Ensure temporal order

            if group_sorted.empty:
                continue

            # Determine biomass type for this experiment
            is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"
            is_recursive = group_sorted["wash_procedure"].iloc[0] == "recursive wash"

            # Get biomass-specific parameters
            C_release = params["C_release_frozen"] if is_frozen else params["C_release_fresh"]
            k_ads = params["k_ads_frozen"] if is_frozen else params["k_ads_fresh"]
            k_des = params["k_des_frozen"] if is_frozen else params["k_des_fresh"]

            # Initialize compartment values for this experiment
            DNA_free = 0.0
            DNA_bound = 0.0

            for idx, row in group_sorted.iterrows():
                step_name = row["process_step"].strip().lower()
                delta_F = row["delta_F"]

                # Determine if this step includes homogenization
                has_homogenization = ("lysis" in step_name or
                                      is_recursive or  # All recursive steps have homogenization
                                      row["cumulative_dose"] > 0)  # Non-zero dose indicates homogenization

                # Set mechanical disruption factor based on step type
                M_factor = params["M_factor_homogenization"] if has_homogenization else params["M_factor_wash"]

                # Process step-specific predictions
                if step_name == "resuspended biomass":
                    if is_frozen:
                        # For frozen biomass, account for initial release due to freeze-thaw
                        F0 = row["F_before"]
                        freeze_thaw_lysis = max(0, 1.0 - F0)
                        DNA_released = C_release * freeze_thaw_lysis

                        # Partition released DNA into free and bound compartments
                        DNA_free = DNA_released * (1 - k_ads)
                        DNA_bound = DNA_released * k_ads
                    else:
                        # For fresh biomass, assume no initial DNA release
                        DNA_free = 0.0
                        DNA_bound = 0.0

                else:  # All other steps (initial lysis, wash steps)
                    # 1. Calculate newly released DNA from lysis (if any)
                    DNA_released = 0.0
                    if pd.notna(delta_F) and delta_F > 0 and pd.notna(C_release):
                        DNA_released = C_release * delta_F

                    # 2. Partition newly released DNA
                    DNA_free_new = DNA_free + DNA_released * (1 - k_ads)
                    DNA_bound_new = DNA_bound + DNA_released * k_ads

                    # 3. Calculate wash efficiency (concentration dependent)
                    W = self._calculate_wash_efficiency(
                        DNA_free_new, params["W_min"], params["W_max"], params["beta"])

                    # 4. Calculate washout and desorption
                    DNA_washed = DNA_free_new * W
                    # Desorption depends on mechanical factor
                    DNA_desorbed = DNA_bound_new * k_des * M_factor

                    # 5. Update compartment values
                    DNA_free = DNA_free_new - DNA_washed + DNA_desorbed
                    DNA_bound = DNA_bound_new - DNA_desorbed

                # Ensure non-negative values
                DNA_free = max(0, DNA_free)
                DNA_bound = max(0, DNA_bound)

                # Prediction for this step is just the free DNA
                df_pred.loc[idx, "dna_pred"] = DNA_free

            # Store final compartment values for debugging
            exp_compartments[exp_id] = {
                "DNA_free": DNA_free,
                "DNA_bound": DNA_bound,
                "biomass_type": "frozen" if is_frozen else "fresh",
                "wash_procedure": "recursive" if is_recursive else "linear"
            }

        # Print compartment info for debugging
        # Uncomment if needed for debugging:
        # for exp_id, comp in exp_compartments.items():
        #     print(f"Exp {exp_id}: {comp['biomass_type']}, {comp['wash_procedure']}, "
        #           f"Final DNA free: {comp['DNA_free']:.2f}, bound: {comp['DNA_bound']:.2f}")

        return df_pred

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the two-compartment model.

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