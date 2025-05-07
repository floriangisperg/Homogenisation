# src/analysis/models/alternative_dna_models.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional, List, Union

from analysis.models.archive.dna_models import DNABaseModel
from analysis.data_processing import calculate_delta_F


class ContinuousDecayDNAModel(DNABaseModel):
    """
    DNA model with continuous decay wash efficiency:
    W(step) = W_base * exp(-λ * step)

    Where:
    - W_base: Base wash efficiency (at step 0)
    - λ: Decay rate parameter
    - step: Wash step number (0-indexed)
    """

    def __init__(self, name="continuous_decay_dna_model"):
        super().__init__(name)
        self.params["W_base"] = np.nan
        self.params["lambda"] = np.nan
        self.initial_guess_W = [0.5, 0.1]  # Initial guess for [W_base, lambda]
        self.bounds_W = [(0.001, 0.999), (0.001, 10.0)]  # Bounds for W_base and lambda
        self.optimization_method_W = "L-BFGS-B"  # Method for W optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, W_base, lambda.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for ContinuousDecayDNAModel fitting: {missing}")
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

            # 2. Fit W parameters
            success_W = self._fit_W_parameters(df_dna_base)

            # Overall success
            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during ContinuousDecayDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_base and lambda parameters.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        # Prepare data for wash efficiency fitting
        df_linear_wash_decay = df_dna_base[
            (df_dna_base["wash_procedure"] == "linear wash") &
            (~df_dna_base["process_step"].str.lower().isin(["resuspended biomass", "initial lysis"]))
            ].copy().dropna(subset=["dna_conc"])

        # Calculate seed DNA concentration (dna_pred_release)
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

        # Add a step number column to the DataFrame
        df_linear_wash_decay = self._add_step_numbers(df_linear_wash_decay)

        try:
            result_W = minimize(
                self._objective_continuous_decay,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_base"] = result_W.x[0]
                self.params["lambda"] = result_W.x[1]
                print(f"W parameters fit successful: W_base={self.params['W_base']:.3f}, "
                      f"lambda={self.params['lambda']:.3f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _add_step_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wash step numbers to the DataFrame.

        Args:
            df: DataFrame with experiment_id and process_step

        Returns:
            DataFrame with 'wash_step_number' column added
        """
        df_with_steps = df.copy()
        df_with_steps['wash_step_number'] = -1  # Default value

        for exp_id in df_with_steps['experiment_id'].unique():
            exp_rows = df_with_steps[df_with_steps['experiment_id'] == exp_id]

            # Sort by index which should be in chronological order
            exp_rows_sorted = exp_rows.sort_index()

            # Assign step numbers (0-indexed)
            step_counter = 0
            for idx in exp_rows_sorted.index:
                df_with_steps.loc[idx, 'wash_step_number'] = step_counter
                step_counter += 1

        return df_with_steps

    def _objective_continuous_decay(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for continuous decay wash efficiency.

        Args:
            params: List of [W_base, lambda] values
            df_linear_wash_decay: DataFrame with linear wash decay data and wash_step_number

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W_base, lambda_decay = params

        if not (0 < W_base < 1 and lambda_decay > 0):
            return np.inf

        sse = 0.0
        total_comparisons = 0

        for exp_id, group in df_linear_wash_decay.groupby("experiment_id"):
            group_sorted = group.sort_index()
            if group_sorted.empty or 'dna_pred_release' not in group_sorted.columns:
                continue

            dna_pred_prev = group_sorted['dna_pred_release'].iloc[0]
            if pd.isna(dna_pred_prev):
                continue

            for idx, row in group_sorted.iterrows():
                step = row['wash_step_number']

                # Calculate step-dependent wash efficiency
                W_current = W_base * np.exp(-lambda_decay * step)

                # Predict DNA concentration for the current step
                dna_pred_current = dna_pred_prev * W_current

                # Calculate SSE contribution if observed data exists
                observed_dna = row["dna_conc"]
                if pd.notna(observed_dna):
                    dna_pred_current_nonneg = max(0, dna_pred_current)
                    sse += (observed_dna - dna_pred_current_nonneg) ** 2
                    total_comparisons += 1

                # Update the 'previous' prediction for the next iteration
                dna_pred_prev = dna_pred_current

        return sse if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the continuous decay wash model.

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

        # Add step numbers for the continuous decay calculation
        df_dna_calc = self._add_step_numbers(df_dna_calc)

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
                    else:  # Subsequent linear wash steps with continuous decay
                        step = row['wash_step_number']

                        # Calculate step-dependent wash efficiency
                        W_current = self.params["W_base"] * np.exp(-self.params["lambda"] * step)

                        # Apply wash decay
                        if pd.notna(dna_pred_prev) and pd.notna(W_current):
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





class SaturationDNAModel(DNABaseModel):
    """
    DNA model with saturation wash efficiency:
    W(step) = W_min + (W_max - W_min) * (1 - exp(-λ * step))

    Where:
    - W_min: Initial wash efficiency
    - W_max: Asymptotic wash efficiency at high step numbers
    - λ: Rate parameter for approach to saturation
    - step: Wash step number (0-indexed)
    """

    def __init__(self, name="saturation_dna_model"):
        super().__init__(name)
        self.params["W_min"] = np.nan
        self.params["W_max"] = np.nan
        self.params["lambda"] = np.nan
        self.initial_guess_W = [0.1, 0.9, 0.5]  # Initial guess for [W_min, W_max, lambda]
        self.bounds_W = [(0.001, 0.5), (0.5, 0.999), (0.001, 10.0)]  # Bounds for parameters
        self.optimization_method_W = "L-BFGS-B"  # Method for W optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, W_min, W_max, lambda.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for SaturationDNAModel fitting: {missing}")
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

            # 2. Fit W parameters
            success_W = self._fit_W_parameters(df_dna_base)

            # Overall success
            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during SaturationDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_min, W_max, and lambda parameters.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        # Prepare data for wash efficiency fitting
        df_linear_wash_decay = df_dna_base[
            (df_dna_base["wash_procedure"] == "linear wash") &
            (~df_dna_base["process_step"].str.lower().isin(["resuspended biomass", "initial lysis"]))
            ].copy().dropna(subset=["dna_conc"])

        # Calculate seed DNA concentration (dna_pred_release)
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

        # Add step numbers
        df_linear_wash_decay = self._add_step_numbers(df_linear_wash_decay)

        try:
            result_W = minimize(
                self._objective_saturation,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_min"] = result_W.x[0]
                self.params["W_max"] = result_W.x[1]
                self.params["lambda"] = result_W.x[2]
                print(f"W parameters fit successful: W_min={self.params['W_min']:.3f}, "
                      f"W_max={self.params['W_max']:.3f}, lambda={self.params['lambda']:.3f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _add_step_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wash step numbers to the DataFrame.

        Args:
            df: DataFrame with experiment_id and process_step

        Returns:
            DataFrame with 'wash_step_number' column added
        """
        df_with_steps = df.copy()
        df_with_steps['wash_step_number'] = -1  # Default value

        for exp_id in df_with_steps['experiment_id'].unique():
            exp_rows = df_with_steps[df_with_steps['experiment_id'] == exp_id]

            # Sort by index which should be in chronological order
            exp_rows_sorted = exp_rows.sort_index()

            # Assign step numbers (0-indexed)
            step_counter = 0
            for idx in exp_rows_sorted.index:
                df_with_steps.loc[idx, 'wash_step_number'] = step_counter
                step_counter += 1

        return df_with_steps

    def _objective_saturation(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for saturation wash efficiency.

        Args:
            params: List of [W_min, W_max, lambda] values
            df_linear_wash_decay: DataFrame with linear wash decay data and wash_step_number

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W_min, W_max, lambda_param = params

        if not (0 < W_min < W_max < 1 and lambda_param > 0):
            return np.inf

        sse = 0.0
        total_comparisons = 0

        for exp_id, group in df_linear_wash_decay.groupby("experiment_id"):
            group_sorted = group.sort_index()
            if group_sorted.empty or 'dna_pred_release' not in group_sorted.columns:
                continue

            dna_pred_prev = group_sorted['dna_pred_release'].iloc[0]
            if pd.isna(dna_pred_prev):
                continue

            for idx, row in group_sorted.iterrows():
                step = row['wash_step_number']

                # Calculate saturation wash efficiency
                W_current = W_min + (W_max - W_min) * (1 - np.exp(-lambda_param * step))

                # Predict DNA concentration for the current step
                dna_pred_current = dna_pred_prev * W_current

                # Calculate SSE contribution if observed data exists
                observed_dna = row["dna_conc"]
                if pd.notna(observed_dna):
                    dna_pred_current_nonneg = max(0, dna_pred_current)
                    sse += (observed_dna - dna_pred_current_nonneg) ** 2
                    total_comparisons += 1

                # Update the 'previous' prediction for the next iteration
                dna_pred_prev = dna_pred_current

        return sse if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the saturation wash model.

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

        # Add step numbers for saturation calculation
        df_dna_calc = self._add_step_numbers(df_dna_calc)

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
                    else:  # Subsequent linear wash steps with saturation efficiency
                        step = row['wash_step_number']

                        # Calculate saturation wash efficiency
                        W_current = self.params["W_min"] + (self.params["W_max"] - self.params["W_min"]) * (
                                    1 - np.exp(-self.params["lambda"] * step))

                        # Apply wash decay
                        if pd.notna(dna_pred_prev) and pd.notna(W_current):
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


class PhysicalAdsorptionDNAModel(DNABaseModel):
    """
    DNA model with physical adsorption wash efficiency:
    W(step) = W_∞ + (W_0 - W_∞) * exp(-k * step)

    Where:
    - W_∞: Asymptotic wash efficiency after many steps
    - W_0: Initial wash efficiency (first step)
    - k: Rate constant for approach to asymptotic value
    - step: Wash step number (0-indexed)
    """

    def __init__(self, name="physical_adsorption_dna_model"):
        super().__init__(name)
        self.params["W_inf"] = np.nan
        self.params["W_0"] = np.nan
        self.params["k"] = np.nan
        self.initial_guess_W = [0.1, 0.9, 0.5]  # Initial guess for [W_inf, W_0, k]
        self.bounds_W = [(0.001, 0.5), (0.5, 0.999), (0.001, 10.0)]  # Bounds for parameters
        self.optimization_method_W = "L-BFGS-B"  # Method for W optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, W_inf, W_0, k.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for PhysicalAdsorptionDNAModel fitting: {missing}")
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

            # 2. Fit W parameters
            success_W = self._fit_W_parameters(df_dna_base)

            # Overall success
            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during PhysicalAdsorptionDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_inf, W_0, and k parameters.

        Args:
            df_dna_base: DataFrame with delta_F and F_before columns

        Returns:
            bool: True if fitting was successful
        """
        # Prepare data for wash efficiency fitting
        df_linear_wash_decay = df_dna_base[
            (df_dna_base["wash_procedure"] == "linear wash") &
            (~df_dna_base["process_step"].str.lower().isin(["resuspended biomass", "initial lysis"]))
            ].copy().dropna(subset=["dna_conc"])

        # Calculate seed DNA concentration (dna_pred_release)
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

        # Add step numbers
        df_linear_wash_decay = self._add_step_numbers(df_linear_wash_decay)

        try:
            result_W = minimize(
                self._objective_physical_adsorption,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_inf"] = result_W.x[0]
                self.params["W_0"] = result_W.x[1]
                self.params["k"] = result_W.x[2]
                print(f"W parameters fit successful: W_inf={self.params['W_inf']:.3f}, "
                      f"W_0={self.params['W_0']:.3f}, k={self.params['k']:.3f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _add_step_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wash step numbers to the DataFrame.

        Args:
            df: DataFrame with experiment_id and process_step

        Returns:
            DataFrame with 'wash_step_number' column added
        """
        df_with_steps = df.copy()
        df_with_steps['wash_step_number'] = -1  # Default value

        for exp_id in df_with_steps['experiment_id'].unique():
            exp_rows = df_with_steps[df_with_steps['experiment_id'] == exp_id]

            # Sort by index which should be in chronological order
            exp_rows_sorted = exp_rows.sort_index()

            # Assign step numbers (0-indexed)
            step_counter = 0
            for idx in exp_rows_sorted.index:
                df_with_steps.loc[idx, 'wash_step_number'] = step_counter
                step_counter += 1

        return df_with_steps

    def _objective_physical_adsorption(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for physical adsorption wash efficiency.

        Args:
            params: List of [W_inf, W_0, k] values
            df_linear_wash_decay: DataFrame with linear wash decay data and wash_step_number

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W_inf, W_0, k = params

        if not (0 < W_inf < W_0 < 1 and k > 0):
            return np.inf

        sse = 0.0
        total_comparisons = 0

        for exp_id, group in df_linear_wash_decay.groupby("experiment_id"):
            group_sorted = group.sort_index()
            if group_sorted.empty or 'dna_pred_release' not in group_sorted.columns:
                continue

            dna_pred_prev = group_sorted['dna_pred_release'].iloc[0]
            if pd.isna(dna_pred_prev):
                continue

            for idx, row in group_sorted.iterrows():
                step = row['wash_step_number']

                # Calculate physical adsorption wash efficiency
                W_current = W_inf + (W_0 - W_inf) * np.exp(-k * step)

                # Predict DNA concentration for the current step
                dna_pred_current = dna_pred_prev * W_current

                # Calculate SSE contribution if observed data exists
                observed_dna = row["dna_conc"]
                if pd.notna(observed_dna):
                    dna_pred_current_nonneg = max(0, dna_pred_current)
                    sse += (observed_dna - dna_pred_current_nonneg) ** 2
                    total_comparisons += 1

                # Update the 'previous' prediction for the next iteration
                dna_pred_prev = dna_pred_current

        return sse if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the physical adsorption wash model.

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

        # Add step numbers for the calculation
        df_dna_calc = self._add_step_numbers(df_dna_calc)

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
                    else:  # Subsequent linear wash steps with physical adsorption efficiency
                        step = row['wash_step_number']

                        # Calculate physical adsorption wash efficiency
                        W_current = self.params["W_inf"] + (self.params["W_0"] - self.params["W_inf"]) * np.exp(
                            -self.params["k"] * step)

                        # Apply wash decay
                        if pd.notna(dna_pred_prev) and pd.notna(W_current):
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