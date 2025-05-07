# src/analysis/models/biomass_specific_dna_model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Any

from analysis.models.dna_models import DNABaseModel
from analysis.data_processing import calculate_delta_F


class BiomassSpecificDNAModel(DNABaseModel):
    """
    DNA model with biomass-specific wash parameters:
    - Separate W_min, W_max for frozen vs. fresh biomass
    - Common beta parameter to reduce overfitting risk

    Wash efficiency formula remains:
    W(C) = W_min + (W_max - W_min) * exp(-β * C)
    """

    def __init__(self, name="biomass_specific_dna_model"):
        super().__init__(name)
        # Replace single parameters with biomass-specific ones
        self.params.pop("W_min", None)
        self.params.pop("W_max", None)

        # Add biomass-specific wash parameters
        self.params.update({
            "W_min_fresh": np.nan,
            "W_max_fresh": np.nan,
            "W_min_frozen": np.nan,
            "W_max_frozen": np.nan,
            "beta": np.nan  # Keep a common beta to reduce overfitting
        })

        # Initial guesses for optimization with regularization
        self.initial_guess_W = [0.05, 0.7, 0.05, 0.7,
                                0.001]  # [W_min_fresh, W_max_fresh, W_min_frozen, W_max_frozen, beta]

        # Bounds for parameters
        self.bounds_W = [
            (0.001, 0.5),  # W_min_fresh - lower bound
            (0.5, 0.999),  # W_max_fresh - upper bound
            (0.001, 0.5),  # W_min_frozen - lower bound
            (0.5, 0.999),  # W_max_frozen - upper bound
            (0.00001, 0.1)  # beta - highly sensitive parameter
        ]

        # Optimization method
        self.optimization_method_W = "L-BFGS-B"

        # Regularization strength - controls how strongly we penalize differences
        # between fresh and frozen parameters (to avoid overfitting)
        self.regularization_lambda = 0.1

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit model parameters to data with biomass-specific wash parameters.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for BiomassSpecificDNAModel fitting: {missing}")
            return False

        try:
            self.training_data = data_with_intact_pred.copy()

            # Calculate delta_F from intact_frac_pred
            df_dna_base = calculate_delta_F(self.training_data)

            # 1. Fit C_release parameters
            success_C = self._fit_C_release_parameters(df_dna_base)
            if not success_C:
                print("Error: Failed to fit C_release parameters")
                return False

            # 2. Fit biomass-specific W parameters
            success_W = self._fit_W_parameters(df_dna_base)

            # Overall success
            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during BiomassSpecificDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_min, W_max, and beta parameters with biomass-specificity.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        # Check if we have both biomass types
        biomass_types = df_dna_base['biomass_type'].unique()
        has_fresh = 'fresh biomass' in biomass_types
        has_frozen = 'frozen biomass' in biomass_types

        if not (has_fresh and has_frozen):
            print(f"Warning: Not all biomass types present. Found: {biomass_types}")
            print("Will try to fit available parameters, but biomass-specific model may not be optimal.")

        # Prepare data for wash efficiency fitting (only linear wash)
        df_linear_wash_decay = df_dna_base[
            (df_dna_base["wash_procedure"] == "linear wash") &
            (~df_dna_base["process_step"].str.lower().isin(["resuspended biomass", "initial lysis"]))
            ].copy().dropna(subset=["dna_conc"])

        # Calculate initial DNA concentration after lysis for each experiment
        dna_pred_release_map = {}
        for exp_id in df_linear_wash_decay["experiment_id"].unique():
            initial_lysis_row = df_dna_base[
                (df_dna_base["experiment_id"] == exp_id) &
                (df_dna_base["process_step"].str.lower() == "initial lysis")
                ]

            if not initial_lysis_row.empty:
                delta_F_init = initial_lysis_row["delta_F"].iloc[0]
                is_frozen_init = initial_lysis_row["biomass_type"].iloc[0] == "frozen biomass"
                C_release_init = self.params["C_release_frozen"] if is_frozen_init else self.params["C_release_fresh"]

                if pd.notna(delta_F_init) and pd.notna(C_release_init):
                    dna_pred_release_map[exp_id] = max(0, C_release_init * delta_F_init)
                else:
                    dna_pred_release_map[exp_id] = np.nan
            else:
                dna_pred_release_map[exp_id] = np.nan

        df_linear_wash_decay['dna_pred_release'] = df_linear_wash_decay['experiment_id'].map(dna_pred_release_map)
        df_linear_wash_decay = df_linear_wash_decay.dropna(subset=['dna_pred_release'])

        if df_linear_wash_decay.empty:
            print("Skipping W parameters fit (insufficient data)")
            return False

        try:
            result_W = minimize(
                self._objective_biomass_specific,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay, self.regularization_lambda),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_min_fresh"] = result_W.x[0]
                self.params["W_max_fresh"] = result_W.x[1]
                self.params["W_min_frozen"] = result_W.x[2]
                self.params["W_max_frozen"] = result_W.x[3]
                self.params["beta"] = result_W.x[4]

                print(f"W parameters fit successful:")
                print(
                    f"  Fresh biomass: W_min={self.params['W_min_fresh']:.3f}, W_max={self.params['W_max_fresh']:.3f}")
                print(
                    f"  Frozen biomass: W_min={self.params['W_min_frozen']:.3f}, W_max={self.params['W_max_frozen']:.3f}")
                print(f"  Common beta={self.params['beta']:.6f}, SSE={result_W.fun:.2f}")

                # Check if frozen and fresh parameters are very different
                w_min_diff = abs(self.params["W_min_fresh"] - self.params["W_min_frozen"])
                w_max_diff = abs(self.params["W_max_fresh"] - self.params["W_max_frozen"])

                if w_min_diff > 0.1 or w_max_diff > 0.1:
                    print(f"Note: Significant differences between fresh and frozen parameters detected.")
                    print(f"  W_min difference: {w_min_diff:.3f}, W_max difference: {w_max_diff:.3f}")
                else:
                    print(
                        "Note: Fresh and frozen parameters are similar, suggesting biomass-specific model may not be necessary.")

                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _objective_biomass_specific(self, params: List[float], df_linear_wash_decay: pd.DataFrame,
                                    reg_lambda: float = 0.1) -> float:
        """
        Objective function for biomass-specific wash efficiency with regularization.

        Args:
            params: List of [W_min_fresh, W_max_fresh, W_min_frozen, W_max_frozen, beta] values
            df_linear_wash_decay: DataFrame with linear wash decay data
            reg_lambda: Regularization strength to penalize large differences between parameters

        Returns:
            float: Regularized sum of squared errors
        """
        W_min_fresh, W_max_fresh, W_min_frozen, W_max_frozen, beta = params

        # Parameter constraints
        if not (0 < W_min_fresh < W_max_fresh < 1 and
                0 < W_min_frozen < W_max_frozen < 1 and
                beta > 0):
            return np.inf

        # Regularization term - penalizes large differences between fresh and frozen parameters
        # This helps prevent overfitting with limited data
        reg_term = reg_lambda * (
                (W_min_fresh - W_min_frozen) ** 2 +
                (W_max_fresh - W_max_frozen) ** 2
        )

        # Data fitting term
        sse = 0.0
        total_comparisons = 0

        for exp_id, group in df_linear_wash_decay.groupby("experiment_id"):
            group_sorted = group.sort_index()
            if group_sorted.empty or 'dna_pred_release' not in group_sorted.columns:
                continue

            # Get initial DNA concentration after lysis
            dna_pred_prev = group_sorted['dna_pred_release'].iloc[0]
            if pd.isna(dna_pred_prev):
                continue

            # Get biomass type for this experiment
            is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"

            # Select appropriate parameters based on biomass type
            W_min = W_min_frozen if is_frozen else W_min_fresh
            W_max = W_max_frozen if is_frozen else W_max_fresh

            for idx, row in group_sorted.iterrows():
                # Calculate biomass-specific concentration-dependent wash efficiency
                W_current = W_min + (W_max - W_min) * np.exp(-beta * dna_pred_prev)

                # Predict DNA concentration for this step
                dna_pred_current = dna_pred_prev * W_current

                # Calculate error contribution if observed data exists
                observed_dna = row["dna_conc"]
                if pd.notna(observed_dna):
                    dna_pred_current_nonneg = max(0, dna_pred_current)
                    sse += (observed_dna - dna_pred_current_nonneg) ** 2
                    total_comparisons += 1

                # Update for next iteration
                dna_pred_prev = dna_pred_current

        # Final objective = data fitting term + regularization term
        objective = sse + reg_term

        return objective if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using biomass-specific parameters.

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

        # Initialize prediction column
        df_pred = df_dna_calc.copy()
        df_pred["dna_pred"] = np.nan

        all_dna_preds = {}

        for exp_id, group in df_pred.groupby("experiment_id"):
            group_sorted = group.sort_index()
            if group_sorted.empty:
                continue

            dna_pred_prev = np.nan
            is_linear = group_sorted["wash_procedure"].iloc[0] == "linear wash"
            is_frozen = group_sorted["biomass_type"].iloc[0] == "frozen biomass"
            C_release = self.params["C_release_frozen"] if is_frozen else self.params["C_release_fresh"]

            # Get biomass-specific wash parameters
            W_min = self.params["W_min_frozen"] if is_frozen else self.params["W_min_fresh"]
            W_max = self.params["W_max_frozen"] if is_frozen else self.params["W_max_fresh"]
            beta = self.params["beta"]  # Common beta parameter

            # Determine F0 for this experiment
            first_row_idx = group_sorted.index[0]
            F0 = group_sorted.loc[first_row_idx, 'F_before']

            group_preds = {}

            for idx, row in group_sorted.iterrows():
                current_pred = np.nan
                step_name = row["process_step"].strip().lower()
                delta_F_current = row["delta_F"]

                # --- Determine prediction based on step and wash type ---
                if step_name == "resuspended biomass":
                    if is_frozen:
                        # Model initial DNA based on freeze-thaw lysis
                        if pd.notna(F0) and pd.notna(self.params["C_release_frozen"]):
                            freeze_thaw_lysis_fraction = max(0, 1.0 - F0)
                            current_pred = self.params["C_release_frozen"] * freeze_thaw_lysis_fraction
                    else:  # Fresh biomass
                        current_pred = 0.0

                elif is_linear:  # Linear Wash Procedure
                    if step_name == "initial lysis":
                        # DNA released is proportional to lysis during this HPH step
                        if pd.notna(delta_F_current) and pd.notna(C_release):
                            current_pred = C_release * delta_F_current
                    else:  # Subsequent linear wash steps with concentration-dependent efficiency
                        if pd.notna(dna_pred_prev):
                            # Calculate biomass-specific concentration-dependent wash efficiency
                            W_current = W_min + (W_max - W_min) * np.exp(-beta * dna_pred_prev)

                            # Apply wash decay
                            current_pred = dna_pred_prev * W_current
                else:  # Recursive Wash Procedure
                    if pd.notna(delta_F_current) and pd.notna(C_release):
                        current_pred = C_release * delta_F_current

                # Store prediction and update previous prediction
                group_preds[idx] = current_pred
                if pd.notna(current_pred):
                    dna_pred_prev = current_pred

            # Update the main dictionary
            all_dna_preds.update(group_preds)

        # Assign predictions to the DataFrame
        df_pred["dna_pred"] = df_pred.index.map(all_dna_preds)

        # Ensure non-negative predictions
        if "dna_pred" in df_pred.columns:
            df_pred["dna_pred"] = df_pred["dna_pred"].clip(lower=0)

        return df_pred