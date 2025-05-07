import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List

from analysis.models.dna_models import DNABaseModel
from ..data_processing import calculate_delta_F

class ConcentrationDependentDNAModel(DNABaseModel):
    """
    DNA model with concentration-dependent wash efficiency:
    W(C) = W_min + (W_max - W_min) * exp(-β * C)

    Where:
    - W_min: Minimum wash efficiency (at high concentrations)
    - W_max: Maximum wash efficiency (at low concentrations)
    - β: Concentration sensitivity parameter
    - C: DNA concentration
    """

    def __init__(self, name="concentration_dependent_dna_model"):
        super().__init__(name)
        self.params["W_min"] = np.nan
        self.params["W_max"] = np.nan
        self.params["beta"] = np.nan
        self.initial_guess_W = [0.1, 0.9, 0.001]  # Initial guess for [W_min, W_max, beta]
        self.bounds_W = [(0.001, 0.5), (0.5, 0.999), (0.00001, 0.1)]  # Bounds for parameters
        self.optimization_method_W = "L-BFGS-B"  # Method for W optimization

    def fit(self, data_with_intact_pred: pd.DataFrame) -> bool:
        """
        Fit all model parameters: C_release_fresh, C_release_frozen, W_min, W_max, beta.

        Args:
            data_with_intact_pred: DataFrame with 'intact_frac_pred' column

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'intact_frac_pred', 'biomass_type',
                         'wash_procedure', 'process_step', 'dna_conc']

        if not all(col in data_with_intact_pred.columns for col in required_cols):
            missing = set(required_cols) - set(data_with_intact_pred.columns)
            print(f"Error: Missing required columns for ConcentrationDependentDNAModel fitting: {missing}")
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
            print(f"Error during ConcentrationDependentDNAModel fitting: {e}")
            traceback.print_exc()
            return False

    def _fit_W_parameters(self, df_dna_base: pd.DataFrame) -> bool:
        """
        Fit the W_min, W_max, and beta parameters.

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
                self._objective_concentration_dependent,
                x0=self.initial_guess_W,
                args=(df_linear_wash_decay,),
                method=self.optimization_method_W,
                bounds=self.bounds_W
            )

            if result_W.success:
                self.params["W_min"] = result_W.x[0]
                self.params["W_max"] = result_W.x[1]
                self.params["beta"] = result_W.x[2]
                print(f"W parameters fit successful: W_min={self.params['W_min']:.3f}, "
                      f"W_max={self.params['W_max']:.3f}, beta={self.params['beta']:.5f}, SSE={result_W.fun:.2f}")
                return True
            else:
                print(f"W parameters fit failed: {result_W.message}")
                return False

        except Exception as e:
            print(f"Error fitting W parameters: {e}")
            return False

    def _objective_concentration_dependent(self, params: List[float], df_linear_wash_decay: pd.DataFrame) -> float:
        """
        Objective function for concentration-dependent wash efficiency.

        Args:
            params: List of [W_min, W_max, beta] values
            df_linear_wash_decay: DataFrame with linear wash decay data

        Returns:
            float: Sum of squared errors between observed and predicted DNA
        """
        W_min, W_max, beta = params

        if not (0 < W_min < W_max < 1 and beta > 0):
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
                # Calculate concentration-dependent wash efficiency
                W_current = W_min + (W_max - W_min) * np.exp(-beta * dna_pred_prev)

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
        Generate DNA concentration predictions using the concentration-dependent wash model.

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
                            # Calculate concentration-dependent wash efficiency
                            W_current = self.params["W_min"] + (self.params["W_max"] - self.params["W_min"]) * np.exp(
                                -self.params["beta"] * dna_pred_prev)

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