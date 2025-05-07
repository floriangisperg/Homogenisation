# src/analysis/models/dna_models.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional, List, Union

from analysis.models.base_model import Model
from analysis.data_processing import calculate_delta_F


class DNABaseModel(Model):
    """
    Base class for DNA concentration models.
    Handles common functionality like parameter initialization and C_release fitting.
    """

    def __init__(self, name="dna_base_model"):
        super().__init__(name)
        self.params = {
            "C_release_fresh": np.nan,
            "C_release_frozen": np.nan
        }
        self.initial_guess_C = [20000]  # Initial guess for C_release parameters
        self.optimization_method_C = "Nelder-Mead"  # Method for C_release optimization
        self.sse = np.nan

    def fit(self, data: pd.DataFrame) -> bool:
        """
        Base implementation for fitting DNA model parameters.
        Implemented by subclasses.
        """
        return False

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Base implementation for DNA prediction.
        Implemented by subclasses.
        """
        return data.copy()

    def _fit_C_release_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit C_release parameters for fresh and frozen biomass.

        Args:
            df_dna_base: DataFrame with 'delta_F' and 'dna_conc' columns

        Returns:
            bool: True if at least one parameter was successfully fitted
        """
        # Filter for recursive wash or initial lysis in linear wash (where release dominates)
        df_dna_train_release = df_dna_base[
            ((df_dna_base["wash_procedure"] == "recursive wash") |
             ((df_dna_base["wash_procedure"] == "linear wash") &
              (df_dna_base["process_step"].str.lower() == "initial lysis")))
            & (df_dna_base["process_step"].str.lower() != "resuspended biomass")
            ].copy().dropna(subset=["dna_conc", "delta_F"])

        success_fresh = success_frozen = False

        # --- For frozen biomass, use the hybrid approach ---
        self.params["C_release_frozen"] = self._estimate_c_release_frozen_hybrid(df_dna_base)
        success_frozen = not np.isnan(self.params["C_release_frozen"])

        # --- For fresh biomass, use the standard approach ---
        df_fresh_release_train = df_dna_train_release[df_dna_train_release["biomass_type"] == "fresh biomass"]
        if not df_fresh_release_train.empty and df_fresh_release_train['delta_F'].sum() > 1e-9:
            try:
                result_C_fresh = minimize(
                    self._objective_dna_release,
                    x0=self.initial_guess_C,
                    args=(df_fresh_release_train, "fresh biomass"),
                    method=self.optimization_method_C
                )
                if result_C_fresh.success:
                    self.params["C_release_fresh"] = max(0, result_C_fresh.x[0])
                    success_fresh = True
                    print(
                        f"C_fresh fit successful: {self.params['C_release_fresh']:.2f}, SSE={result_C_fresh.fun:.2f}")
            except Exception as e:
                print(f"Error fitting C_fresh: {e}")
        else:
            print("Skipping C_fresh fit (insufficient data)")

        # Return True if at least one parameter was successfully fitted
        return success_fresh or success_frozen

    def _objective_dna_release(self, params: List[float], df_dna_model: pd.DataFrame,
                               biomass_type_filter: str) -> float:
        """
        Objective function for C_release parameter optimization.

        Args:
            params: List with C_release value
            df_dna_model: DataFrame with DNA data
            biomass_type_filter: Which biomass type to filter for

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        C_release = params[0]
        if C_release < 0:
            return np.inf  # Constraint

        df_subset = df_dna_model[df_dna_model["biomass_type"] == biomass_type_filter].copy()
        if df_subset.empty:
            return 1e12

        df_subset["dna_pred"] = C_release * df_subset["delta_F"]

        valid_idx = df_subset["dna_conc"].notna() & df_subset["dna_pred"].notna()
        if not valid_idx.any():
            return 1e12

        observed = df_subset.loc[valid_idx, "dna_conc"]
        predicted = df_subset.loc[valid_idx, "dna_pred"]
        residuals = observed - predicted

        return np.sum(residuals ** 2)

    def _prepare_data_for_prediction(self, df_dna_calc: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare base data for DNA prediction, including calculating delta_F.

        Args:
            df_dna_calc: DataFrame with intact_frac_pred, etc.

        Returns:
            DataFrame with calculated intermediate values
        """
        required_cols = ['experiment_id', 'biomass_type', 'wash_procedure',
                         'process_step', 'intact_frac_pred']

        if not all(col in df_dna_calc.columns for col in required_cols):
            missing = set(required_cols) - set(df_dna_calc.columns)
            raise ValueError(f"Missing required columns for DNA prediction preparation: {missing}")

        return df_dna_calc.copy()

    def _estimate_c_release_frozen_hybrid(self, df_train: pd.DataFrame, weight_fresh: float = 0.3) -> float:
        """
        Hybrid approach to estimate C_release_frozen using both frozen and fresh biomass data.

        Args:
            df_train: Training data DataFrame
            weight_fresh: Weight given to the estimate from fresh biomass (0-1)

        Returns:
            Estimated C_release_frozen value
        """
        # Split data by biomass type
        df_frozen = df_train[df_train["biomass_type"] == "frozen biomass"].copy()
        df_fresh = df_train[df_train["biomass_type"] == "fresh biomass"].copy()

        # Check if we have both types
        has_frozen = not df_frozen.empty
        has_fresh = not df_fresh.empty

        if not has_frozen and not has_fresh:
            print("Warning: No valid biomass data for hybrid C_release estimation")
            return 5000.0  # Fallback to a reasonable default

        # 1. Direct estimate from frozen biomass if available
        c_frozen_direct = None
        if has_frozen:
            # Filter for release data (recursive wash or initial lysis)
            df_frozen_release = df_frozen[
                ((df_frozen["wash_procedure"] == "recursive wash") |
                 ((df_frozen["wash_procedure"] == "linear wash") &
                  (df_frozen["process_step"].str.lower() == "initial lysis")))
                & (df_frozen["process_step"].str.lower() != "resuspended biomass")
                ].copy().dropna(subset=["dna_conc", "delta_F"])

            if not df_frozen_release.empty and df_frozen_release['delta_F'].sum() > 1e-9:
                # Use minimize to find optimal C_release_frozen
                result = minimize(
                    self._objective_dna_release,
                    x0=self.initial_guess_C,
                    args=(df_frozen_release, "frozen biomass"),
                    method=self.optimization_method_C
                )
                if result.success:
                    c_frozen_direct = max(0, result.x[0])
                    print(f"Direct frozen estimate: C_release_frozen = {c_frozen_direct:.2f}")

        # 2. Indirect estimate from fresh biomass if available
        c_frozen_indirect = None
        if has_fresh:
            # Filter for release data (recursive wash or initial lysis)
            df_fresh_release = df_fresh[
                ((df_fresh["wash_procedure"] == "recursive wash") |
                 ((df_fresh["wash_procedure"] == "linear wash") &
                  (df_fresh["process_step"].str.lower() == "initial lysis")))
                & (df_fresh["process_step"].str.lower() != "resuspended biomass")
                ].copy().dropna(subset=["dna_conc", "delta_F"])

            if not df_fresh_release.empty and df_fresh_release['delta_F'].sum() > 1e-9:
                # Use minimize to find optimal C_release_fresh
                result = minimize(
                    self._objective_dna_release,
                    x0=self.initial_guess_C,
                    args=(df_fresh_release, "fresh biomass"),
                    method=self.optimization_method_C
                )
                if result.success:
                    c_fresh = max(0, result.x[0])
                    # Apply transfer function from fresh to frozen (based on your data)
                    # Empirical relationship: C_frozen ≈ 0.42 * C_fresh
                    c_frozen_indirect = 0.42 * c_fresh
                    print(
                        f"Indirect estimate: C_release_fresh = {c_fresh:.2f}, implied C_release_frozen = {c_frozen_indirect:.2f}")

        # 3. Combine estimates
        if c_frozen_direct is not None and c_frozen_indirect is not None:
            # Weighted combination
            c_frozen_combined = (1 - weight_fresh) * c_frozen_direct + weight_fresh * c_frozen_indirect
            print(f"Hybrid estimate: C_release_frozen = {c_frozen_combined:.2f} (weight_fresh={weight_fresh})")
            return c_frozen_combined
        elif c_frozen_direct is not None:
            print(f"Using direct estimate only: C_release_frozen = {c_frozen_direct:.2f}")
            return c_frozen_direct
        elif c_frozen_indirect is not None:
            print(f"Using indirect estimate only: C_release_frozen = {c_frozen_indirect:.2f}")
            return c_frozen_indirect
        else:
            print("Warning: Failed to estimate C_release_frozen, using default")
            return 5000.0  # Fallback


class SingleWashDNAModel(DNABaseModel):
    """
    DNA model with a single wash efficiency parameter (W_wash).
    """

    def __init__(self, name="single_wash_dna_model"):
        super().__init__(name)
        self.params["W_wash"] = np.nan
        self.initial_guess_W = [0.5]  # Initial guess for W_wash
        self.bounds_W = [(0.001, 0.999)]  # Bounds for W_wash
        self.optimization_method_W = "L-BFGS-B"  # Method for W_wash optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, and W_wash.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for SingleWashDNAModel fitting: {missing}")
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

            # 2. Fit W_wash parameter
            success_W = self._fit_W_parameter(df_dna_base)

            # Overall success
            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            import traceback
            print(f"Error during SingleWashDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameter(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_wash parameter for the single wash model.

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
            print("Skipping W_wash fit (insufficient data)")
            return False

        try:
            result_W = minimize(
                self._objective_dna_wash,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_wash"] = result_W.x[0]
                print(f"W_wash fit successful: {self.params['W_wash']:.3f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W_wash fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W_wash parameter: {e}")
            return False

    def _objective_dna_wash(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for wash efficiency factor W.

        Args:
            params: List with W_wash value
            df_linear_wash_decay: DataFrame with linear wash decay data

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W = params[0]
        if not (0 <= W <= 1):
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
                dna_pred_current = dna_pred_prev * W
                observed_dna = row["dna_conc"]

                if pd.notna(observed_dna):
                    sse += (observed_dna - dna_pred_current) ** 2
                    total_comparisons += 1

                dna_pred_prev = dna_pred_current

        return sse if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the single wash model.

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
                        # Model initial DNA based on freeze-thaw lysis (1 - F0)
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
                    else:  # Subsequent linear wash steps
                        # DNA concentration decays based on previous step's concentration and W
                        if pd.notna(dna_pred_prev) and pd.notna(self.params["W_wash"]):
                            current_pred = dna_pred_prev * self.params["W_wash"]
                else:  # Recursive Wash Procedure
                    if pd.notna(delta_F_current) and pd.notna(C_release):
                        current_pred = C_release * delta_F_current

                # Store prediction and update previous prediction
                group_preds[idx] = current_pred
                dna_pred_prev = current_pred

            # Update the main dictionary
            all_dna_preds.update(group_preds)

        # Assign predictions to the DataFrame
        df_pred["dna_pred"] = df_pred.index.map(all_dna_preds)

        # Ensure non-negative predictions
        if "dna_pred" in df_pred.columns:
            df_pred["dna_pred"] = df_pred["dna_pred"].clip(lower=0)

        return df_pred


class StepDependentWashDNAModel(DNABaseModel):
    """
    DNA model with step-dependent wash efficiency parameters (W_wash_1st, W_wash_subsequent).
    """

    def __init__(self, name="step_dependent_wash_dna_model"):
        super().__init__(name)
        self.params["W_wash_1st"] = np.nan
        self.params["W_wash_subsequent"] = np.nan
        self.initial_guess_W = [0.5, 0.5]  # Initial guess for [W_1st, W_subsequent]
        self.bounds_W = [(0.001, 0.999), (0.001, 0.999)]  # Bounds for both W parameters
        self.optimization_method_W = "L-BFGS-B"  # Method for W optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, W_wash_1st, W_wash_subsequent.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for StepDependentWashDNAModel fitting: {missing}")
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
            print(f"Error during StepDependentWashDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_wash_1st and W_wash_subsequent parameters.

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

        try:
            result_W = minimize(
                self._objective_dna_wash_step_dependent,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_wash_1st"] = result_W.x[0]
                self.params["W_wash_subsequent"] = result_W.x[1]
                print(f"W parameters fit successful: W_1st={self.params['W_wash_1st']:.3f}, "
                      f"W_sub={self.params['W_wash_subsequent']:.3f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _objective_dna_wash_step_dependent(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for step-dependent wash efficiency factors.

        Args:
            params: List with [W_1st, W_subsequent] values
            df_linear_wash_decay: DataFrame with linear wash decay data

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W_1st, W_subsequent = params

        if not (0 <= W_1st <= 1 and 0 <= W_subsequent <= 1):
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

            is_first_wash_step = True

            for idx, row in group_sorted.iterrows():
                # Determine which W parameter to use
                W_current = W_1st if is_first_wash_step else W_subsequent

                # Predict DNA concentration
                dna_pred_current = dna_pred_prev * W_current

                # Calculate SSE contribution
                observed_dna = row["dna_conc"]
                if pd.notna(observed_dna):
                    dna_pred_current_nonneg = max(0, dna_pred_current)
                    sse += (observed_dna - dna_pred_current_nonneg) ** 2
                    total_comparisons += 1

                # Update for next iteration
                dna_pred_prev = dna_pred_current
                is_first_wash_step = False

        return sse if total_comparisons > 0 else 1e12

    def predict(self, data_with_intact_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Generate DNA concentration predictions using the step-dependent wash model.

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

            # Determine F0 for this experiment
            first_row_idx = group_sorted.index[0]
            F0 = group_sorted.loc[first_row_idx, 'F_before']

            group_preds = {}
            is_first_linear_wash_step_encountered = False

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
                        is_first_linear_wash_step_encountered = False
                    else:  # Subsequent linear wash steps with step-dependent decay
                        # Determine which W parameter to use
                        if not is_first_linear_wash_step_encountered:
                            W_current = self.params["W_wash_1st"]
                            is_first_linear_wash_step_encountered = True
                        else:
                            W_current = self.params["W_wash_subsequent"]

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