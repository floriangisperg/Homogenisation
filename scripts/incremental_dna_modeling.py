# scripts/incremental_dna_modeling.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
import time
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional, Union

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths configuration
DATA_DIR = PROJECT_ROOT / "data" / "scFv"
DATA_FILENAME = "scfv_lysis.xlsx"
RESULTS_DIR = PROJECT_ROOT / "results" / "incremental_dna_modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Data Processing
# =====================================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from Excel file and standardize columns."""
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Standardize column names
    column_mapping = {
        "Process step": "process_step",
        "total number of passages at 650 bar": "total_passages_650",
        "total number of passages at 1000 bar": "total_passages_1000",
        "Intact biomass percentage [%]": "intact_biomass_percent",
        "DNA [ng/µL]": "dna_conc",
        "DNA std. dev. [ng/µL]": "dna_std_dev",
        "wash procedure": "wash_procedure",
        "biomass type": "biomass_type",
        "experiment id": "experiment_id"
    }

    # Rename columns that exist in the DataFrame
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=columns_to_rename)

    # Convert numeric columns
    numeric_cols = ["total_passages_650", "total_passages_1000", "intact_biomass_percent",
                    "dna_conc", "dna_std_dev"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Process categorical columns
    categorical_cols = ["process_step", "wash_procedure", "biomass_type"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip().str.lower()

    # Calculate observed_frac
    if 'intact_biomass_percent' in df.columns:
        df["observed_frac"] = df["intact_biomass_percent"] / 100.0

    logging.info(f"Processed data: {len(df)} rows")

    return df


def fit_intact_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit the intact fraction model (F = F0 * exp(-k * P^alpha * N)).
    This function is simplified and uses predefined k and alpha values.

    Returns:
        Tuple of (DataFrame with intact_frac_pred, Parameters dictionary)
    """
    logging.info("Fitting intact fraction model...")

    # For simplicity, we're using fixed parameters based on previous analysis
    # In a full implementation, you would fit these parameters
    k = 1e-6
    alpha = 1.5

    # Add intact_frac_pred column
    df_with_pred = df.copy()
    df_with_pred['intact_frac_pred'] = np.nan

    # Calculate cumulative dose for each experiment
    for exp_id, group in df.groupby('experiment_id'):
        # Get process steps in order
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
        biomass_type = group_sorted['biomass_type'].iloc[0]

        # Determine F0 based on biomass type
        if biomass_type == 'fresh biomass':
            F0 = 1.0  # Fresh biomass starts fully intact
        else:
            # Frozen biomass - get initial intact fraction
            if 'intact_biomass_percent' in group_sorted.columns:
                F0_row = group_sorted[group_sorted['process_step'] == 'resuspended biomass']
                if not F0_row.empty and pd.notna(F0_row['intact_biomass_percent'].iloc[0]):
                    F0 = F0_row['intact_biomass_percent'].iloc[0] / 100.0
                else:
                    # Default if not available
                    F0 = 0.8
            else:
                F0 = 0.8

        # Calculate cumulative dose and intact fraction for each step
        cumulative_dose = 0

        for idx, row in group_sorted.iterrows():
            # Calculate pressure term for this step
            p_term_650 = 0
            p_term_1000 = 0

            if pd.notna(row['total_passages_650']) and row['total_passages_650'] > 0:
                p_term_650 = (650 ** alpha) * row['total_passages_650']

            if pd.notna(row['total_passages_1000']) and row['total_passages_1000'] > 0:
                p_term_1000 = (1000 ** alpha) * row['total_passages_1000']

            cumulative_dose = k * (p_term_650 + p_term_1000)

            # Calculate intact fraction
            intact_frac = F0 * np.exp(-cumulative_dose)

            # Update DataFrame
            df_with_pred.loc[idx, 'intact_frac_pred'] = intact_frac
            df_with_pred.loc[idx, 'cumulative_dose'] = cumulative_dose

    intact_params = {'k': k, 'alpha': alpha}
    logging.info(f"Intact model parameters: k={k:.2e}, alpha={alpha:.2f}")

    return df_with_pred, intact_params


# =====================================================================
# Base Model Class
# =====================================================================

class DNAModel:
    """Base class for DNA concentration models."""

    def __init__(self, name: str = "dna_model"):
        self.name = name
        self.params = {
            "C_release_fresh": np.nan,  # DNA release coefficient for fresh biomass
            "C_release_frozen": np.nan  # DNA release coefficient for frozen biomass
        }
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> bool:
        """
        Fit the model parameters to data.

        Args:
            data: DataFrame with 'intact_frac_pred', 'dna_conc', etc.

        Returns:
            bool: True if fitting was successful
        """
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with the fitted model.

        Args:
            data: DataFrame with required columns

        Returns:
            DataFrame with 'dna_pred' column added
        """
        pass

    def has_required_params(self) -> bool:
        """Check if all parameters are set."""
        return all(pd.notna(value) for value in self.params.values())

    def get_params(self) -> Dict[str, float]:
        """Get a copy of the model parameters."""
        return self.params.copy()


# =====================================================================
# Model 1: Basic DNA Model
# =====================================================================

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
            logging.error(f"Missing required columns for BasicDNAModel")
            return False

        try:
            # 1. First fit C_release parameters
            success_C = self._fit_release_parameters(data)

            # 2. Then fit wash efficiency parameter
            success_W = self._fit_wash_parameter(data)

            self.fitted = success_C and success_W
            return self.fitted

        except Exception as e:
            logging.error(f"Error fitting BasicDNAModel: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _fit_release_parameters(self, data: pd.DataFrame) -> bool:
        """Fit C_release parameters based on initial lysis steps."""
        logging.info("Fitting DNA release parameters...")

        # Filter for initial lysis data points
        initial_lysis = data[data['process_step'] == 'initial lysis'].copy()

        if initial_lysis.empty:
            logging.warning("No initial lysis data found")
            return False

        # Separate by biomass type
        fresh_data = initial_lysis[initial_lysis['biomass_type'] == 'fresh biomass']
        frozen_data = initial_lysis[initial_lysis['biomass_type'] == 'frozen biomass']

        # For fresh biomass: C_release * delta_F ≈ dna_conc
        if not fresh_data.empty:
            fresh_data = fresh_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not fresh_data.empty:
                # Get previous intact fraction (from resuspended biomass or default to 1.0)
                fresh_data['prev_intact'] = 1.0
                for idx, row in fresh_data.iterrows():
                    exp_id = row['experiment_id']
                    resuspended = data[(data['experiment_id'] == exp_id) &
                                       (data['process_step'] == 'resuspended biomass')]
                    if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                        fresh_data.loc[idx, 'prev_intact'] = resuspended['intact_frac_pred'].iloc[0]

                # Calculate delta_F = prev_intact - intact_frac_pred
                fresh_data['delta_F'] = fresh_data['prev_intact'] - fresh_data['intact_frac_pred']

                # Simple estimation: C_release = dna_conc / delta_F
                valid_data = fresh_data[fresh_data['delta_F'] > 0.01].copy()
                if not valid_data.empty:
                    C_vals = valid_data['dna_conc'] / valid_data['delta_F']
                    self.params["C_release_fresh"] = C_vals.mean()
                    logging.info(f"C_release_fresh = {self.params['C_release_fresh']:.2f}")

        # For frozen biomass: similar approach
        if not frozen_data.empty:
            frozen_data = frozen_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not frozen_data.empty:
                # Get previous intact fraction
                frozen_data['prev_intact'] = 0.8  # Default if not available
                for idx, row in frozen_data.iterrows():
                    exp_id = row['experiment_id']
                    resuspended = data[(data['experiment_id'] == exp_id) &
                                       (data['process_step'] == 'resuspended biomass')]
                    if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                        frozen_data.loc[idx, 'prev_intact'] = resuspended['intact_frac_pred'].iloc[0]

                # Calculate delta_F
                frozen_data['delta_F'] = frozen_data['prev_intact'] - frozen_data['intact_frac_pred']

                # Estimate C_release
                valid_data = frozen_data[frozen_data['delta_F'] > 0.01].copy()
                if not valid_data.empty:
                    C_vals = valid_data['dna_conc'] / valid_data['delta_F']
                    self.params["C_release_frozen"] = C_vals.mean()
                    logging.info(f"C_release_frozen = {self.params['C_release_frozen']:.2f}")

        success = (pd.notna(self.params["C_release_fresh"]) or
                   pd.notna(self.params["C_release_frozen"]))

        if not success:
            logging.warning("Failed to fit any C_release parameters")

        return success

    def _fit_wash_parameter(self, data: pd.DataFrame) -> bool:
        """Fit wash efficiency parameter from wash steps."""
        logging.info("Fitting wash efficiency parameter...")

        # Filter for wash steps with DNA data
        wash_steps = data[data['process_step'].str.contains('wash') &
                          ~data['process_step'].str.contains('resuspended')].copy()
        wash_steps = wash_steps.dropna(subset=['dna_conc'])

        if wash_steps.empty:
            logging.warning("No wash steps with DNA data found")
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
            wash_indices = df_pred.index[df_pred['process_step'].str.contains('wash') &
                                         ~df_pred['process_step'].str.contains('resuspended')]
            valid_idx = df_pred.loc[wash_indices, 'dna_conc'].notna() & df_pred.loc[wash_indices, 'dna_pred'].notna()

            if not any(valid_idx):
                return np.inf

            errors = df_pred.loc[wash_indices[valid_idx], 'dna_conc'] - df_pred.loc[wash_indices[valid_idx], 'dna_pred']
            return np.sum(errors ** 2)

        # Optimize
        try:
            result = minimize(
                objective,
                x0=[0.5],  # Initial guess
                bounds=[(0.01, 0.99)],  # W must be between 0 and 1
                method='L-BFGS-B'
            )

            if result.success:
                self.params["wash_efficiency"] = result.x[0]
                logging.info(f"Wash efficiency = {self.params['wash_efficiency']:.4f}")
                return True
            else:
                logging.warning(f"Wash parameter optimization failed: {result.message}")
                return False
        except Exception as e:
            logging.error(f"Error during wash parameter optimization: {e}")
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

            if pd.isna(C_release):
                # Skip if we don't have parameters for this biomass type
                continue

            # Process each step
            dna_prev = 0

            for idx, row in group_sorted.iterrows():
                step = row['process_step']

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
                            resuspended = group_sorted[group_sorted['process_step'] == 'resuspended biomass']
                            if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                                prev_frac = resuspended['intact_frac_pred'].iloc[0]

                        delta_F = prev_frac - row['intact_frac_pred']
                        dna_pred = C_release * delta_F
                    else:
                        dna_pred = np.nan

                elif 'wash' in step:
                    # Apply wash efficiency to previous DNA concentration
                    W = params["wash_efficiency"]
                    if pd.notna(W) and pd.notna(dna_prev):
                        dna_pred = dna_prev * (1 - W)
                    else:
                        dna_pred = np.nan

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
            logging.warning("Not all parameters are defined")

        return self._predict_with_params(data, self.params)


# =====================================================================
# Model 2: Step-Dependent Wash Model
# =====================================================================

class StepDependentWashModel(BasicDNAModel):
    """
    Extends the basic model with different wash efficiencies for:
    1. First wash after initial lysis
    2. Subsequent washes

    This model tests the hypothesis that wash efficiency changes across steps.
    """

    def __init__(self, name: str = "step_dependent_wash_model"):
        super().__init__(name)
        # Replace single wash parameter with two step-dependent parameters
        self.params.pop("wash_efficiency", None)
        self.params["first_wash_efficiency"] = np.nan
        self.params["subsequent_wash_efficiency"] = np.nan

    def _fit_wash_parameter(self, data: pd.DataFrame) -> bool:
        """Fit step-dependent wash efficiency parameters."""
        logging.info("Fitting step-dependent wash parameters...")

        # Filter for wash steps with DNA data
        wash_steps = data[data['process_step'].str.contains('wash') &
                          ~data['process_step'].str.contains('resuspended')].copy()
        wash_steps = wash_steps.dropna(subset=['dna_conc'])

        if wash_steps.empty:
            logging.warning("No wash steps with DNA data found")
            return False

        # Objective function for two wash parameters
        def objective(params):
            W1, W2 = params
            if not (0 < W1 < 1 and 0 < W2 < 1):
                return np.inf

            # Make predictions with these W values
            test_params = self.params.copy()
            test_params["first_wash_efficiency"] = W1
            test_params["subsequent_wash_efficiency"] = W2

            df_pred = self._predict_with_params(data, test_params)

            # Calculate error on wash steps only
            wash_indices = df_pred.index[df_pred['process_step'].str.contains('wash') &
                                         ~df_pred['process_step'].str.contains('resuspended')]
            valid_idx = df_pred.loc[wash_indices, 'dna_conc'].notna() & df_pred.loc[wash_indices, 'dna_pred'].notna()

            if not any(valid_idx):
                return np.inf

            errors = df_pred.loc[wash_indices[valid_idx], 'dna_conc'] - df_pred.loc[wash_indices[valid_idx], 'dna_pred']
            return np.sum(errors ** 2)

        # Optimize
        try:
            result = minimize(
                objective,
                x0=[0.6, 0.4],  # Initial guess: first wash more efficient
                bounds=[(0.01, 0.99), (0.01, 0.99)],  # Both Ws between 0 and 1
                method='L-BFGS-B'
            )

            if result.success:
                self.params["first_wash_efficiency"] = result.x[0]
                self.params["subsequent_wash_efficiency"] = result.x[1]
                logging.info(f"First wash efficiency = {self.params['first_wash_efficiency']:.4f}")
                logging.info(f"Subsequent wash efficiency = {self.params['subsequent_wash_efficiency']:.4f}")
                return True
            else:
                logging.warning(f"Wash parameter optimization failed: {result.message}")
                return False
        except Exception as e:
            logging.error(f"Error during wash parameter optimization: {e}")
            return False

    def _predict_with_params(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Generate predictions using specified parameters with step-dependent wash."""
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

            if pd.isna(C_release):
                # Skip if we don't have parameters for this biomass type
                continue

            # Process each step
            dna_prev = 0
            wash_count = 0

            for idx, row in group_sorted.iterrows():
                step = row['process_step']

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
                            resuspended = group_sorted[group_sorted['process_step'] == 'resuspended biomass']
                            if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                                prev_frac = resuspended['intact_frac_pred'].iloc[0]

                        delta_F = prev_frac - row['intact_frac_pred']
                        dna_pred = C_release * delta_F
                    else:
                        dna_pred = np.nan

                    # Reset wash counter
                    wash_count = 0

                elif 'wash' in step:
                    # Increment wash counter
                    wash_count += 1

                    # Apply appropriate wash efficiency based on step
                    if wash_count == 1:
                        W = params["first_wash_efficiency"]
                    else:
                        W = params["subsequent_wash_efficiency"]

                    if pd.notna(W) and pd.notna(dna_prev):
                        dna_pred = dna_prev * (1 - W)
                    else:
                        dna_pred = np.nan

                else:
                    dna_pred = np.nan

                # Update prediction for this row
                df_pred.loc[idx, 'dna_pred'] = dna_pred

                # Update previous DNA concentration for next step
                if pd.notna(dna_pred):
                    dna_prev = dna_pred

        return df_pred


# =====================================================================
# Model 3: Process-Type-Dependent Model
# =====================================================================

class ProcessTypeDependentModel(StepDependentWashModel):
    """
    Extends the step-dependent model with different parameters for:
    1. Linear wash process
    2. Recursive wash process

    This model tests the hypothesis that different wash procedures behave differently.
    """

    def __init__(self, name: str = "process_type_dependent_model"):
        super().__init__(name)
        # Replace wash parameters with process-specific versions
        self.params.pop("first_wash_efficiency", None)
        self.params.pop("subsequent_wash_efficiency", None)

        # Linear wash parameters
        self.params["linear_first_wash_efficiency"] = np.nan
        self.params["linear_subsequent_wash_efficiency"] = np.nan

        # Recursive wash parameters
        self.params["recursive_first_wash_efficiency"] = np.nan
        self.params["recursive_subsequent_wash_efficiency"] = np.nan

    def _fit_wash_parameter(self, data: pd.DataFrame) -> bool:
        """Fit process-type-dependent wash parameters."""
        logging.info("Fitting process-type-dependent wash parameters...")

        # Filter for wash steps with DNA data
        wash_steps = data[data['process_step'].str.contains('wash') &
                          ~data['process_step'].str.contains('resuspended')].copy()
        wash_steps = wash_steps.dropna(subset=['dna_conc'])

        if wash_steps.empty:
            logging.warning("No wash steps with DNA data found")
            return False

        # Check if we have both linear and recursive wash data
        has_linear = any(wash_steps['wash_procedure'] == 'linear wash')
        has_recursive = any(wash_steps['wash_procedure'] == 'recursive wash')

        logging.info(f"Data available - Linear wash: {has_linear}, Recursive wash: {has_recursive}")

        # Objective function for four wash parameters
        def objective(params):
            L1, L2, R1, R2 = params
            if not (0 < L1 < 1 and 0 < L2 < 1 and 0 < R1 < 1 and 0 < R2 < 1):
                return np.inf

            # Make predictions with these parameters
            test_params = self.params.copy()
            test_params["linear_first_wash_efficiency"] = L1
            test_params["linear_subsequent_wash_efficiency"] = L2
            test_params["recursive_first_wash_efficiency"] = R1
            test_params["recursive_subsequent_wash_efficiency"] = R2

            df_pred = self._predict_with_params(data, test_params)

            # Calculate error on wash steps only
            wash_indices = df_pred.index[df_pred['process_step'].str.contains('wash') &
                                         ~df_pred['process_step'].str.contains('resuspended')]
            valid_idx = df_pred.loc[wash_indices, 'dna_conc'].notna() & df_pred.loc[wash_indices, 'dna_pred'].notna()

            if not any(valid_idx):
                return np.inf

            errors = df_pred.loc[wash_indices[valid_idx], 'dna_conc'] - df_pred.loc[wash_indices[valid_idx], 'dna_pred']
            return np.sum(errors ** 2)

        # Optimize
        try:
            result = minimize(
                objective,
                x0=[0.7, 0.5, 0.4, 0.2],  # Initial guess
                bounds=[(0.01, 0.99)] * 4,  # All parameters between 0 and 1
                method='L-BFGS-B'
            )

            if result.success:
                self.params["linear_first_wash_efficiency"] = result.x[0]
                self.params["linear_subsequent_wash_efficiency"] = result.x[1]
                self.params["recursive_first_wash_efficiency"] = result.x[2]
                self.params["recursive_subsequent_wash_efficiency"] = result.x[3]

                logging.info(f"Linear first wash efficiency = {self.params['linear_first_wash_efficiency']:.4f}")
                logging.info(
                    f"Linear subsequent wash efficiency = {self.params['linear_subsequent_wash_efficiency']:.4f}")
                logging.info(f"Recursive first wash efficiency = {self.params['recursive_first_wash_efficiency']:.4f}")
                logging.info(
                    f"Recursive subsequent wash efficiency = {self.params['recursive_subsequent_wash_efficiency']:.4f}")
                return True
            else:
                logging.warning(f"Wash parameter optimization failed: {result.message}")
                return False
        except Exception as e:
            logging.error(f"Error during wash parameter optimization: {e}")
            return False

    def _predict_with_params(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Generate predictions using process-type-dependent parameters."""
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

            # Get biomass type and wash procedure
            is_frozen = group_sorted['biomass_type'].iloc[0] == 'frozen biomass'
            is_recursive = group_sorted['wash_procedure'].iloc[0] == 'recursive wash'

            C_release = params["C_release_frozen"] if is_frozen else params["C_release_fresh"]

            if pd.isna(C_release):
                # Skip if we don't have parameters for this biomass type
                continue

            # Process each step
            dna_prev = 0
            wash_count = 0

            for idx, row in group_sorted.iterrows():
                step = row['process_step']

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
                            resuspended = group_sorted[group_sorted['process_step'] == 'resuspended biomass']
                            if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                                prev_frac = resuspended['intact_frac_pred'].iloc[0]

                        delta_F = prev_frac - row['intact_frac_pred']
                        dna_pred = C_release * delta_F
                    else:
                        dna_pred = np.nan

                    # Reset wash counter
                    wash_count = 0

                elif 'wash' in step:
                    # Increment wash counter
                    wash_count += 1

                    # Select appropriate wash efficiency based on process type and step
                    if is_recursive:
                        if wash_count == 1:
                            W = params["recursive_first_wash_efficiency"]
                        else:
                            W = params["recursive_subsequent_wash_efficiency"]
                    else:  # Linear wash
                        if wash_count == 1:
                            W = params["linear_first_wash_efficiency"]
                        else:
                            W = params["linear_subsequent_wash_efficiency"]

                    if pd.notna(W) and pd.notna(dna_prev):
                        dna_pred = dna_prev * (1 - W)
                    else:
                        dna_pred = np.nan

                else:
                    dna_pred = np.nan

                # Update prediction for this row
                df_pred.loc[idx, 'dna_pred'] = dna_pred

                # Update previous DNA concentration for next step
                if pd.notna(dna_pred):
                    dna_prev = dna_pred

        return df_pred


# =====================================================================
# Model 4: Two-Compartment Model
# =====================================================================

class TwoCompartmentModel(DNAModel):
    """
    A two-compartment model that separates DNA into free and bound fractions.
    It accounts for:
    1. DNA release from cell lysis into both free and bound compartments
    2. Different adsorption/release behavior based on biomass type
    3. Desorption of bound DNA during processing
    4. Process-type dependent parameters

    This tests the hypothesis that DNA binding and desorption mechanisms
    are responsible for the difference between recursive and linear wash.
    """

    def __init__(self, name: str = "two_compartment_model"):
        super().__init__(name)

        # Binding parameters
        self.params["adsorption_fraction_fresh"] = np.nan  # Fraction of released DNA that gets bound
        self.params["adsorption_fraction_frozen"] = np.nan

        # Wash parameters (vary by process type)
        self.params["linear_wash_efficiency"] = np.nan  # Efficiency of DNA removal from free compartment
        self.params["recursive_wash_efficiency"] = np.nan

        # Desorption parameters
        self.params["linear_desorption_rate"] = np.nan  # Rate of bound DNA moving to free compartment
        self.params["recursive_desorption_rate"] = np.nan

    def fit(self, data: pd.DataFrame) -> bool:
        """Fit the two-compartment model parameters."""
        required_cols = ['experiment_id', 'biomass_type', 'process_step',
                         'wash_procedure', 'intact_frac_pred', 'dna_conc']

        if not all(col in data.columns for col in required_cols):
            logging.error(f"Missing required columns for TwoCompartmentModel")
            return False

        try:
            # 1. First fit C_release parameters
            success_C = self._fit_release_parameters(data)

            # 2. Then fit compartment parameters
            success_comp = self._fit_compartment_parameters(data)

            self.fitted = success_C and success_comp
            return self.fitted

        except Exception as e:
            logging.error(f"Error fitting TwoCompartmentModel: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _fit_release_parameters(self, data: pd.DataFrame) -> bool:
        """Fit C_release parameters based on initial lysis steps."""
        logging.info("Fitting DNA release parameters...")

        # Filter for initial lysis data points
        initial_lysis = data[data['process_step'] == 'initial lysis'].copy()

        if initial_lysis.empty:
            logging.warning("No initial lysis data found")
            return False

        # Separate by biomass type
        fresh_data = initial_lysis[initial_lysis['biomass_type'] == 'fresh biomass']
        frozen_data = initial_lysis[initial_lysis['biomass_type'] == 'frozen biomass']

        # For fresh biomass: C_release * delta_F ≈ dna_conc
        if not fresh_data.empty:
            fresh_data = fresh_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not fresh_data.empty:
                # Get previous intact fraction (from resuspended biomass or default to 1.0)
                fresh_data['prev_intact'] = 1.0
                for idx, row in fresh_data.iterrows():
                    exp_id = row['experiment_id']
                    resuspended = data[(data['experiment_id'] == exp_id) &
                                       (data['process_step'] == 'resuspended biomass')]
                    if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                        fresh_data.loc[idx, 'prev_intact'] = resuspended['intact_frac_pred'].iloc[0]

                # Calculate delta_F = prev_intact - intact_frac_pred
                fresh_data['delta_F'] = fresh_data['prev_intact'] - fresh_data['intact_frac_pred']

                # Simple estimation: C_release = dna_conc / delta_F
                valid_data = fresh_data[fresh_data['delta_F'] > 0.01].copy()
                if not valid_data.empty:
                    C_vals = valid_data['dna_conc'] / valid_data['delta_F']
                    self.params["C_release_fresh"] = C_vals.mean()
                    logging.info(f"C_release_fresh = {self.params['C_release_fresh']:.2f}")

        # For frozen biomass: similar approach
        if not frozen_data.empty:
            frozen_data = frozen_data.dropna(subset=['dna_conc', 'intact_frac_pred'])
            if not frozen_data.empty:
                # Get previous intact fraction
                frozen_data['prev_intact'] = 0.8  # Default if not available
                for idx, row in frozen_data.iterrows():
                    exp_id = row['experiment_id']
                    resuspended = data[(data['experiment_id'] == exp_id) &
                                       (data['process_step'] == 'resuspended biomass')]
                    if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                        frozen_data.loc[idx, 'prev_intact'] = resuspended['intact_frac_pred'].iloc[0]

                # Calculate delta_F
                frozen_data['delta_F'] = frozen_data['prev_intact'] - frozen_data['intact_frac_pred']

                # Estimate C_release
                valid_data = frozen_data[frozen_data['delta_F'] > 0.01].copy()
                if not valid_data.empty:
                    C_vals = valid_data['dna_conc'] / valid_data['delta_F']
                    self.params["C_release_frozen"] = C_vals.mean()
                    logging.info(f"C_release_frozen = {self.params['C_release_frozen']:.2f}")

        success = (pd.notna(self.params["C_release_fresh"]) or
                   pd.notna(self.params["C_release_frozen"]))

        if not success:
            logging.warning("Failed to fit any C_release parameters")

        return success

    def _fit_compartment_parameters(self, data: pd.DataFrame) -> bool:
        """Fit two-compartment model parameters."""
        logging.info("Fitting two-compartment model parameters...")

        # Check if we have data for both process types
        has_linear = any(data['wash_procedure'] == 'linear wash')
        has_recursive = any(data['wash_procedure'] == 'recursive wash')

        logging.info(f"Data available - Linear wash: {has_linear}, Recursive wash: {has_recursive}")

        # Objective function for compartment parameters
        def objective(params):
            # Unpack parameters:
            # 0: adsorption_fraction_fresh
            # 1: adsorption_fraction_frozen
            # 2: linear_wash_efficiency
            # 3: recursive_wash_efficiency
            # 4: linear_desorption_rate
            # 5: recursive_desorption_rate

            if not all(0 < p < 1 for p in params):
                return np.inf

            # Make predictions with these parameters
            test_params = self.params.copy()
            test_params["adsorption_fraction_fresh"] = params[0]
            test_params["adsorption_fraction_frozen"] = params[1]
            test_params["linear_wash_efficiency"] = params[2]
            test_params["recursive_wash_efficiency"] = params[3]
            test_params["linear_desorption_rate"] = params[4]
            test_params["recursive_desorption_rate"] = params[5]

            # Generate predictions
            df_pred = self._predict_with_params(data, test_params)

            # Calculate error using all valid data points
            valid_idx = df_pred['dna_conc'].notna() & df_pred['dna_pred'].notna()

            if not any(valid_idx):
                return np.inf

            errors = df_pred.loc[valid_idx, 'dna_conc'] - df_pred.loc[valid_idx, 'dna_pred']
            return np.sum(errors ** 2)

        # Optimize
        try:
            # Initial parameter guesses based on hypotheses
            initial_guess = [
                0.3,  # adsorption_fraction_fresh - 30% gets bound
                0.5,  # adsorption_fraction_frozen - 50% gets bound (higher due to damaged cells)
                0.7,  # linear_wash_efficiency - 70% removal in linear wash
                0.3,  # recursive_wash_efficiency - 30% removal in recursive (mechanical disruption re-releases DNA)
                0.1,  # linear_desorption_rate - 10% desorption per step in linear
                0.4  # recursive_desorption_rate - 40% desorption per step in recursive (higher due to mechanical)
            ]

            bounds = [(0.01, 0.99)] * 6  # All parameters between 0 and 1

            result = minimize(
                objective,
                x0=initial_guess,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success:
                self.params["adsorption_fraction_fresh"] = result.x[0]
                self.params["adsorption_fraction_frozen"] = result.x[1]
                self.params["linear_wash_efficiency"] = result.x[2]
                self.params["recursive_wash_efficiency"] = result.x[3]
                self.params["linear_desorption_rate"] = result.x[4]
                self.params["recursive_desorption_rate"] = result.x[5]

                logging.info(f"Adsorption fraction (fresh) = {self.params['adsorption_fraction_fresh']:.4f}")
                logging.info(f"Adsorption fraction (frozen) = {self.params['adsorption_fraction_frozen']:.4f}")
                logging.info(f"Linear wash efficiency = {self.params['linear_wash_efficiency']:.4f}")
                logging.info(f"Recursive wash efficiency = {self.params['recursive_wash_efficiency']:.4f}")
                logging.info(f"Linear desorption rate = {self.params['linear_desorption_rate']:.4f}")
                logging.info(f"Recursive desorption rate = {self.params['recursive_desorption_rate']:.4f}")
                return True
            else:
                logging.warning(f"Compartment parameter optimization failed: {result.message}")
                return False
        except Exception as e:
            logging.error(f"Error during compartment parameter optimization: {e}")
            return False

    def _predict_with_params(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        """Generate predictions using two-compartment model parameters."""
        df_pred = data.copy()
        df_pred['dna_pred'] = np.nan

        # Optional: add debugging columns to show compartment values
        df_pred['dna_free'] = np.nan
        df_pred['dna_bound'] = np.nan

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

            # Get biomass type and wash procedure
            is_frozen = group_sorted['biomass_type'].iloc[0] == 'frozen biomass'
            is_recursive = group_sorted['wash_procedure'].iloc[0] == 'recursive wash'

            # Get appropriate parameters
            C_release = params["C_release_frozen"] if is_frozen else params["C_release_fresh"]
            adsorption_fraction = params["adsorption_fraction_frozen"] if is_frozen else params[
                "adsorption_fraction_fresh"]
            wash_efficiency = params["recursive_wash_efficiency"] if is_recursive else params["linear_wash_efficiency"]
            desorption_rate = params["recursive_desorption_rate"] if is_recursive else params["linear_desorption_rate"]

            if pd.isna(C_release) or pd.isna(adsorption_fraction) or pd.isna(wash_efficiency) or pd.isna(
                    desorption_rate):
                # Skip if we don't have necessary parameters
                continue

            # Initialize compartment values
            DNA_free = 0.0
            DNA_bound = 0.0

            for idx, row in group_sorted.iterrows():
                step = row['process_step']

                if step == 'resuspended biomass':
                    # For frozen biomass, estimate initial DNA release
                    if is_frozen:
                        # Assume some DNA already released from freeze-thaw damage
                        initial_frac = row['intact_frac_pred'] if pd.notna(row['intact_frac_pred']) else 0.8
                        freeze_thaw_release = 1 - initial_frac

                        # Partition released DNA between free and bound compartments
                        DNA_free = C_release * freeze_thaw_release * (1 - adsorption_fraction)
                        DNA_bound = C_release * freeze_thaw_release * adsorption_fraction
                    else:
                        # For fresh biomass, no initial DNA release
                        DNA_free = 0.0
                        DNA_bound = 0.0

                elif step == 'initial lysis':
                    # DNA release proportional to cell lysis
                    if pd.notna(row['intact_frac_pred']):
                        prev_frac = 1.0  # Assuming initial state is 100% intact
                        if is_frozen:
                            # For frozen, use the actual initial intact fraction
                            resuspended = group_sorted[group_sorted['process_step'] == 'resuspended biomass']
                            if not resuspended.empty and pd.notna(resuspended['intact_frac_pred'].iloc[0]):
                                prev_frac = resuspended['intact_frac_pred'].iloc[0]

                        delta_F = prev_frac - row['intact_frac_pred']

                        # Newly released DNA
                        new_DNA = C_release * delta_F

                        # Partition new DNA between compartments
                        DNA_free += new_DNA * (1 - adsorption_fraction)
                        DNA_bound += new_DNA * adsorption_fraction

                elif 'wash' in step:
                    # Two-compartment processes during wash:

                    # 1. Desorption: Move DNA from bound to free compartment
                    desorption_amount = DNA_bound * desorption_rate
                    DNA_bound -= desorption_amount
                    DNA_free += desorption_amount

                    # 2. For recursive wash, additional DNA release from lysis
                    if is_recursive and pd.notna(row['intact_frac_pred']):
                        # Find previous intact fraction
                        prev_idx = group_sorted.index[group_sorted['step_order'] < row['step_order']]
                        if len(prev_idx) > 0:
                            prev_step = group_sorted.loc[prev_idx[-1]]
                            if pd.notna(prev_step['intact_frac_pred']):
                                # Calculate additional lysis
                                delta_F = prev_step['intact_frac_pred'] - row['intact_frac_pred']
                                if delta_F > 0:
                                    # New DNA release
                                    new_DNA = C_release * delta_F
                                    # Partition
                                    DNA_free += new_DNA * (1 - adsorption_fraction)
                                    DNA_bound += new_DNA * adsorption_fraction

                    # 3. Washing: Remove DNA from free compartment
                    DNA_free *= (1 - wash_efficiency)

                else:
                    # Unrecognized step - do nothing
                    pass

                # Update prediction and compartment values for this row
                df_pred.loc[idx, 'dna_pred'] = DNA_free
                df_pred.loc[idx, 'dna_free'] = DNA_free
                df_pred.loc[idx, 'dna_bound'] = DNA_bound

        return df_pred

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with the fitted model."""
        if not self.has_required_params():
            logging.warning("Not all parameters are defined")

        return self._predict_with_params(data, self.params)


# =====================================================================
# Model Evaluation
# =====================================================================

def evaluate_model(model: DNAModel, data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate a model's performance.

    Args:
        model: Fitted DNA model
        data: DataFrame with ground truth

    Returns:
        Dictionary of metrics
    """
    # Generate predictions
    df_pred = model.predict(data)

    # Filter out rows without DNA data
    valid_idx = df_pred['dna_conc'].notna() & df_pred['dna_pred'].notna()
    valid_data = df_pred[valid_idx]

    if len(valid_data) == 0:
        logging.warning("No valid data points for evaluation")
        return {
            "name": model.name,
            "R²": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "valid_points": 0
        }

    # Calculate metrics
    try:
        r2 = r2_score(valid_data['dna_conc'], valid_data['dna_pred'])
        rmse = np.sqrt(mean_squared_error(valid_data['dna_conc'], valid_data['dna_pred']))
        mae = mean_absolute_error(valid_data['dna_conc'], valid_data['dna_pred'])

        metrics = {
            "name": model.name,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "valid_points": len(valid_data)
        }

        logging.info(f"Model: {model.name}")
        logging.info(f"  R²: {r2:.4f}")
        logging.info(f"  RMSE: {rmse:.2f}")
        logging.info(f"  MAE: {mae:.2f}")
        logging.info(f"  Valid points: {len(valid_data)}")

        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return {
            "name": model.name,
            "R²": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "valid_points": len(valid_data),
            "error": str(e)
        }


def run_loocv(model_class, data: pd.DataFrame, intact_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run Leave-One-Out Cross-Validation.

    Args:
        model_class: Class of the model to evaluate
        data: DataFrame with all data
        intact_data: DataFrame with intact_frac_pred

    Returns:
        Dictionary with LOOCV results
    """
    model_name = model_class().name
    logging.info(f"Running LOOCV for {model_name}...")

    # Get unique experiment IDs
    exp_ids = data['experiment_id'].unique()
    n_folds = len(exp_ids)

    if n_folds < 2:
        logging.error("Need at least 2 different experiment IDs for LOOCV")
        return {"error": "Insufficient experiments for LOOCV"}

    # Initialize containers for results
    fold_metrics = []
    all_predictions = []

    # Run LOOCV
    for fold, hold_out_id in enumerate(exp_ids):
        logging.info(f"Fold {fold + 1}/{n_folds}: Holding out Experiment {hold_out_id}")

        # Split data
        train_data = intact_data[intact_data['experiment_id'] != hold_out_id].copy()
        test_data = intact_data[intact_data['experiment_id'] == hold_out_id].copy()

        if train_data.empty or test_data.empty:
            logging.warning(f"Skipping fold {fold + 1}: Empty train or test set")
            continue

        # Create and fit model
        model = model_class()
        success = model.fit(train_data)

        if not success:
            logging.warning(f"Skipping fold {fold + 1}: Model fitting failed")
            continue

        # Generate predictions for test data
        test_pred = model.predict(test_data)

        # Evaluate predictions
        valid_idx = test_pred['dna_conc'].notna() & test_pred['dna_pred'].notna()
        valid_test = test_pred[valid_idx]

        if len(valid_test) == 0:
            logging.warning(f"Skipping fold {fold + 1}: No valid test predictions")
            continue

        # Calculate metrics
        try:
            r2 = r2_score(valid_test['dna_conc'], valid_test['dna_pred'])
            rmse = np.sqrt(mean_squared_error(valid_test['dna_conc'], valid_test['dna_pred']))
            mae = mean_absolute_error(valid_test['dna_conc'], valid_test['dna_pred'])

            fold_metric = {
                "fold": fold + 1,
                "experiment_id": hold_out_id,
                "R²": r2,
                "RMSE": rmse,
                "MAE": mae,
                "valid_points": len(valid_test),
                "parameters": model.get_params()
            }

            fold_metrics.append(fold_metric)
            all_predictions.append(test_pred)

            logging.info(f"Fold {fold + 1} complete: R² = {r2:.4f}, RMSE = {rmse:.2f}")
        except Exception as e:
            logging.error(f"Error calculating metrics for fold {fold + 1}: {e}")

    # Combine all predictions
    combined_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Calculate overall metrics
    overall_metrics = {}
    if not combined_predictions.empty:
        valid_idx = combined_predictions['dna_conc'].notna() & combined_predictions['dna_pred'].notna()
        valid_data = combined_predictions[valid_idx]

        if not valid_data.empty:
            overall_metrics = {
                "R²": r2_score(valid_data['dna_conc'], valid_data['dna_pred']),
                "RMSE": np.sqrt(mean_squared_error(valid_data['dna_conc'], valid_data['dna_pred'])),
                "MAE": mean_absolute_error(valid_data['dna_conc'], valid_data['dna_pred']),
                "valid_points": len(valid_data)
            }

    # Calculate average of fold metrics
    avg_metrics = {}
    for metric in ["R²", "RMSE", "MAE"]:
        values = [fold[metric] for fold in fold_metrics if metric in fold]
        if values:
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            avg_metrics[f"std_{metric}"] = np.std(values)

    # Compile results
    loocv_results = {
        "model_name": model_name,
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "overall_metrics": overall_metrics,
        "avg_metrics": avg_metrics,
        "predictions": combined_predictions
    }

    # Log summary
    logging.info(f"LOOCV Summary for {model_name}:")
    for metric, value in overall_metrics.items():
        logging.info(f"  Overall {metric}: {value:.4f}")

    for metric, value in avg_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    return loocv_results


# =====================================================================
# Visualization Functions
# =====================================================================

def plot_parity(model_name: str, predictions: pd.DataFrame, output_dir: Path):
    """
    Create parity plot for DNA concentration predictions.

    Args:
        model_name: Name of the model
        predictions: DataFrame with 'dna_conc' and 'dna_pred'
        output_dir: Directory to save plot
    """
    valid_idx = (predictions['dna_conc'] > 0) & (predictions['dna_pred'] > 0)
    valid_data = predictions[valid_idx]

    if valid_data.empty:
        logging.warning(f"No valid data for parity plot of {model_name}")
        return

    plt.figure(figsize=(8, 8))

    # Add experiment_id as color if available
    if 'experiment_id' in valid_data.columns:
        plt.scatter(valid_data['dna_conc'], valid_data['dna_pred'],
                    c=valid_data['experiment_id'], cmap='tab10', alpha=0.7)
    else:
        plt.scatter(valid_data['dna_conc'], valid_data['dna_pred'], alpha=0.7)

    # Draw parity line
    min_val = min(valid_data['dna_conc'].min(), valid_data['dna_pred'].min()) * 0.8
    max_val = max(valid_data['dna_conc'].max(), valid_data['dna_pred'].max()) * 1.2
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Observed DNA Concentration [ng/µL]')
    plt.ylabel('Predicted DNA Concentration [ng/µL]')
    plt.title(f'{model_name}: Observed vs. Predicted DNA Concentration')

    # Add R² value if we can calculate it
    if len(valid_data) > 1:
        r2 = r2_score(valid_data['dna_conc'], valid_data['dna_pred'])
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    # Save plot
    output_file = output_dir / f"{model_name.lower().replace(' ', '_')}_parity_plot.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logging.info(f"Saved parity plot to {output_file}")


def plot_predictions_by_step(model_name: str, predictions: pd.DataFrame, output_dir: Path):
    """
    Plot predictions vs. observed data by process step.

    Args:
        model_name: Name of the model
        predictions: DataFrame with predictions
        output_dir: Directory to save plots
    """
    # Create directory for experiment plots
    exp_plot_dir = output_dir / f"{model_name.lower().replace(' ', '_')}_experiments"
    exp_plot_dir.mkdir(exist_ok=True)

    # Get unique experiment IDs
    exp_ids = predictions['experiment_id'].unique()

    for exp_id in exp_ids:
        exp_data = predictions[predictions['experiment_id'] == exp_id].copy()

        # Skip if no DNA data
        if exp_data['dna_conc'].isna().all() or exp_data['dna_pred'].isna().all():
            continue

        # Sort by step
        step_order = {
            'resuspended biomass': 0,
            'initial lysis': 1,
            '1st wash': 2,
            '2nd wash': 3,
            '3rd wash': 4,
            '4th wash': 5
        }

        exp_data['step_order'] = exp_data['process_step'].map(lambda x: step_order.get(x, 999))
        exp_data = exp_data.sort_values('step_order')

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot observed and predicted values
        plt.plot(exp_data['process_step'], exp_data['dna_conc'], 'o-', label='Observed')
        plt.plot(exp_data['process_step'], exp_data['dna_pred'], 'x--', label='Predicted')

        # Add experiment info
        biomass_type = exp_data['biomass_type'].iloc[0]
        wash_procedure = exp_data['wash_procedure'].iloc[0]

        # Set y-axis to log scale
        plt.yscale('log')

        # Rotate x-labels for readability
        plt.xticks(rotation=45, ha='right')

        # Add labels and title
        plt.xlabel('Process Step')
        plt.ylabel('DNA Concentration [ng/µL]')
        plt.title(f'Experiment {exp_id}: {biomass_type}, {wash_procedure}')
        plt.legend()

        # Add grid
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

        # Save plot
        plt.tight_layout()
        output_file = exp_plot_dir / f"experiment_{exp_id}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

    logging.info(f"Saved experiment plots to {exp_plot_dir}")


def plot_model_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """
    Create bar charts comparing model performance metrics.

    Args:
        results: List of model result dictionaries
        output_dir: Directory to save plots
    """
    # Extract data for plotting
    model_names = [result['model_name'] for result in results]
    r2_values = [result['overall_metrics'].get('R²', np.nan) for result in results]
    rmse_values = [result['overall_metrics'].get('RMSE', np.nan) for result in results]
    mae_values = [result['overall_metrics'].get('MAE', np.nan) for result in results]

    # Create directory for comparison plots
    comparison_dir = output_dir / "model_comparison"
    comparison_dir.mkdir(exist_ok=True)

    # Plot R² comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, r2_values)

    # Add value labels
    for bar, value in zip(bars, r2_values):
        if not np.isnan(value):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{value:.4f}',
                ha='center',
                va='bottom'
            )

    plt.xlabel('Model')
    plt.ylabel('R²')
    plt.title('Model Comparison: R²')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(comparison_dir / "r2_comparison.png", dpi=300)
    plt.close()

    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, rmse_values)

    for bar, value in zip(bars, rmse_values):
        if not np.isnan(value):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{value:.2f}',
                ha='center',
                va='bottom'
            )

    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Comparison: RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(comparison_dir / "rmse_comparison.png", dpi=300)
    plt.close()

    # Plot MAE comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, mae_values)

    for bar, value in zip(bars, mae_values):
        if not np.isnan(value):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{value:.2f}',
                ha='center',
                va='bottom'
            )

    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.title('Model Comparison: MAE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(comparison_dir / "mae_comparison.png", dpi=300)
    plt.close()

    logging.info(f"Saved model comparison plots to {comparison_dir}")


# =====================================================================
# Main Function
# =====================================================================

def main():
    """Main function to run the incremental DNA modeling analysis."""
    start_time = time.time()
    logging.info("=== Incremental DNA Modeling Analysis ===")

    # Make sure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = load_data(DATA_DIR / DATA_FILENAME)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Fit intact fraction model
    df_with_intact, intact_params = fit_intact_model(df)

    # Initialize and test models
    models = [
        BasicDNAModel(name="Basic DNA Model"),
        StepDependentWashModel(name="Step-Dependent Wash Model"),
        ProcessTypeDependentModel(name="Process-Type-Dependent Model"),
        TwoCompartmentModel(name="Two-Compartment Model")
    ]

    # Results container
    model_results = []

    # Test each model with LOOCV
    for model_class in [m.__class__ for m in models]:
        loocv_result = run_loocv(model_class, df, df_with_intact)
        model_results.append(loocv_result)

        # Save predictions
        if 'predictions' in loocv_result and not loocv_result['predictions'].empty:
            predictions = loocv_result['predictions']
            model_name = loocv_result['model_name']

            # Create visualizations
            plot_parity(model_name, predictions, RESULTS_DIR)
            plot_predictions_by_step(model_name, predictions, RESULTS_DIR)

            # Save predictions to CSV
            pred_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv"
            predictions.to_csv(pred_file, index=False)
            logging.info(f"Saved predictions to {pred_file}")

            # Save LOOCV results to JSON
            import json

            # Create a version without the large predictions DataFrame for JSON
            json_result = {k: v for k, v in loocv_result.items() if k != 'predictions'}

            # Convert numpy values to Python types
            def convert_np(obj):
                if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_np(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_np(item) for item in obj]
                else:
                    return obj

            json_result = convert_np(json_result)

            json_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_loocv_results.json"
            with open(json_file, 'w') as f:
                json.dump(json_result, f, indent=2)
            logging.info(f"Saved LOOCV results to {json_file}")

    # Create model comparison plots
    plot_model_comparison(model_results, RESULTS_DIR)

    # Create aggregated summary report
    report_file = RESULTS_DIR / "model_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("DNA Model Comparison Report\n")
        f.write("=========================\n\n")

        # Write summary table
        f.write(f"{'Model':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10}\n")
        f.write("-" * 60 + "\n")

        for result in model_results:
            model_name = result['model_name']
            metrics = result['overall_metrics']

            r2 = metrics.get('R²', np.nan)
            rmse = metrics.get('RMSE', np.nan)
            mae = metrics.get('MAE', np.nan)

            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            rmse_str = f"{rmse:.2f}" if not np.isnan(rmse) else "N/A"
            mae_str = f"{mae:.2f}" if not np.isnan(mae) else "N/A"

            f.write(f"{model_name:<30} {r2_str:<10} {rmse_str:<10} {mae_str:<10}\n")

        f.write("\n\n")

        # Write detailed fold results
        f.write("Detailed Results by Fold\n")
        f.write("=======================\n\n")

        for result in model_results:
            model_name = result['model_name']
            f.write(f"{model_name}\n")
            f.write("-" * len(model_name) + "\n\n")

            f.write(f"{'Experiment':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}\n")
            f.write("-" * 45 + "\n")

            for fold in result.get('fold_metrics', []):
                exp_id = fold.get('experiment_id', 'Unknown')
                r2 = fold.get('R²', np.nan)
                rmse = fold.get('RMSE', np.nan)
                mae = fold.get('MAE', np.nan)

                r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                rmse_str = f"{rmse:.2f}" if not np.isnan(rmse) else "N/A"
                mae_str = f"{mae:.2f}" if not np.isnan(mae) else "N/A"

                f.write(f"{exp_id:<15} {r2_str:<10} {rmse_str:<10} {mae_str:<10}\n")

            # Write average across folds
            avg_metrics = result.get('avg_metrics', {})
            avg_r2 = avg_metrics.get('avg_R²', np.nan)
            avg_rmse = avg_metrics.get('avg_RMSE', np.nan)
            avg_mae = avg_metrics.get('avg_MAE', np.nan)

            r2_str = f"{avg_r2:.4f}" if not np.isnan(avg_r2) else "N/A"
            rmse_str = f"{avg_rmse:.2f}" if not np.isnan(avg_rmse) else "N/A"
            mae_str = f"{avg_mae:.2f}" if not np.isnan(avg_mae) else "N/A"

            f.write("-" * 45 + "\n")
            f.write(f"{'Average':<15} {r2_str:<10} {rmse_str:<10} {mae_str:<10}\n")
            f.write("\n\n")

    logging.info(f"Saved model comparison report to {report_file}")

    # Print execution time
    end_time = time.time()
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()