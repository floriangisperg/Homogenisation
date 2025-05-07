# src/analysis/models/simple_dna_model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, List

from .structure import DNAModel


class BasicDNAModel(DNAModel):
    """
    Simplest DNA model with three core concepts:
    1. DNA release proportional to cell lysis
    2. Different coefficients for fresh/frozen biomass
    3. Simple wash efficiency that removes a fraction of DNA
    """

    def __init__(self, name: str = "basic_dna_model"):
        super().__init__(name)
        # Add wash efficiency parameter
        self.params["wash_efficiency"] = np.nan

    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit the model parameters to data.

        Args:
            data: DataFrame with 'intact_frac_pred', 'dna_conc', etc.

        Returns:
            bool: True if fitting was successful
        """
        required_cols = ['experiment_id', 'biomass_type', 'process_step',
                         'intact_frac_pred', 'dna_conc']

        if not all(col in data.columns for col in required_cols):
            print(f"Missing required columns for BasicDNAModel")
            return False

        try:
            # 1. First fit C_release parameters
            success_C = self._fit_release_parameters(data)

            # 2. Then fit wash efficiency parameter
            success_W = self._fit_wash_parameter(data)

            return success_C and success_W

        except Exception as e:
            print(f"Error fitting BasicDNAModel: {e}")
            return False

    def _fit_release_parameters(self, data: pd.DataFrame) -> bool:
        """Fit C_release parameters based on initial lysis steps."""
        # Filter for initial lysis data points
        initial_lysis = data[data['process_step'].str.lower() == 'initial lysis'].copy()

        if initial_lysis.empty:
            print("No initial lysis data found")
            return False

        # Separate by biomass type
        fresh_data = initial_lysis[initial_lysis['biomass_type'] == 'fresh biomass']
        frozen_data = initial_lysis[initial_lysis['biomass_type'] == 'frozen biomass']

        # For fresh biomass: C_release * delta_F ≈ dna_conc
        if not fresh_data.empty:
            fresh_data = fresh_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not fresh_data.empty:
                # Calculate delta_F = 1 - intact_frac_pred
                fresh_data['delta_F'] = 1 - fresh_data['intact_frac_pred']

                # Simple estimation: C_release = dna_conc / delta_F
                C_vals = fresh_data['dna_conc'] / fresh_data['delta_F']
                self.params["C_release_fresh"] = C_vals.mean()

        # For frozen biomass: similar approach
        if not frozen_data.empty:
            frozen_data = frozen_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not frozen_data.empty:
                frozen_data['delta_F'] = 1 - frozen_data['intact_frac_pred']
                C_vals = frozen_data['dna_conc'] / frozen_data['delta_F']
                self.params["C_release_frozen"] = C_vals.mean()

        return (pd.notna(self.params["C_release_fresh"]) or
                pd.notna(self.params["C_release_frozen"]))

    def _fit_wash_parameter(self, data: pd.DataFrame) -> bool:
        """Fit wash efficiency parameter from wash steps."""
        # Filter for wash steps
        wash_steps = data[data['process_step'].str.contains('wash')].copy()

        if wash_steps.empty:
            print("No wash steps found")
            return False

        # Simple objective function: minimize sum of squared errors
        def objective(params):
            W = params[0]
            if not (0 < W < 1):
                return np.inf

            # Make predictions with this W value
            test_params = self.params.copy()
            test_params["wash_efficiency"] = W

            df_pred = self._predict_with_params(data, test_params)

            # Calculate error on wash steps only
            wash_indices = df_pred.index[df_pred['process_step'].str.contains('wash')]
            errors = df_pred.loc[wash_indices, 'dna_conc'] - df_pred.loc[wash_indices, 'dna_pred']
            return np.sum(errors ** 2)

        # Optimize
        result = minimize(
            objective,
            x0=[0.5],  # Initial guess
            bounds=[(0.01, 0.99)]  # W must be between 0 and 1
        )

        if result.success:
            self.params["wash_efficiency"] = result.x[0]
            return True
        else:
            print(f"Wash parameter optimization failed: {result.message}")
            return False

    def _predict_with_params(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Generate predictions using specified parameters."""
        df_pred = data.copy()
        df_pred['dna_pred'] = np.nan

        for exp_id, group in df_pred.groupby('experiment_id'):
            # Sort by process step order
            step_order = {
                'resuspended biomass': 0,
                'initial lysis': 1,
                '1st wash': 2,
                '2nd wash': 3,
                '3rd wash': 4,
                '4th wash': 5
            }

            # Add step order and sort
            group['step_order'] = group['process_step'].map(
                lambda x: step_order.get(x.lower(), 999))
            group_sorted = group.sort_values('step_order')

            # Get biomass type
            is_frozen = group_sorted['biomass_type'].iloc[0] == 'frozen biomass'
            C_release = params["C_release_frozen"] if is_frozen else params["C_release_fresh"]

            # Process each step
            dna_prev = 0

            for idx, row in group_sorted.iterrows():
                step = row['process_step'].lower()

                if step == 'resuspended biomass':
                    # For frozen biomass, estimate initial DNA release
                    if is_frozen:
                        # Assume some DNA already released from freeze-thaw damage
                        initial_frac = row['intact_frac_pred'] if pd.notna(row['intact_frac_pred']) else 0.8
                        freeze_thaw_release = 1 - initial_frac
                        dna_pred = C_release * freeze_thaw_release
                    else:
                        # For fresh biomass, no initial DNA release
                        dna_pred = 0

                elif step == 'initial lysis':
                    # DNA release proportional to cell lysis
                    if pd.notna(row['intact_frac_pred']):
                        prev_frac = 1.0  # Assuming initial state is 100% intact
                        if is_frozen:
                            # For frozen, use the actual initial intact fraction
                            resuspended = group_sorted[
                                group_sorted['process_step'].str.lower() == 'resuspended biomass']
                            if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                                prev_frac = resuspended['intact_frac_pred'].iloc[0]

                        delta_F = prev_frac - row['intact_frac_pred']
                        dna_pred = C_release * delta_F
                    else:
                        dna_pred = np.nan

                elif 'wash' in step:
                    # Apply wash efficiency to previous DNA concentration
                    W = params["wash_efficiency"]
                    dna_pred = dna_prev * (1 - W)

                else:
                    dna_pred = np.nan

                # Update prediction for this row
                df_pred.loc[idx, 'dna_pred'] = dna_pred

                # Update previous DNA concentration for next step
                if pd.notna(dna_pred):
                    dna_prev = dna_pred

        return df_pred

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with the fitted model."""
        if not self.has_required_params():
            print("Warning: Not all parameters are defined")

        return self._predict_with_params(data, self.params)