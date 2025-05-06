# src/analysis/cross_validation.py
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List, Union, Type

from .models.base_model import Model
from .models.intact_model import IntactModel
from .models.dna_models import DNABaseModel
from .evaluation import calculate_metrics, calculate_dna_metrics
from .plotting import plot_parity, plot_dna_parity_matplotlib


class CrossValidator:
    """
    Implements Leave-One-Out Cross-Validation (LOOCV) for lysis and DNA models.
    """

    def __init__(self,
                 output_dir: Path,
                 intact_model_class: Type[Model] = IntactModel,
                 dna_model_class: Type[Model] = None):
        """
        Initialize the cross-validator.

        Args:
            output_dir: Directory to save results and plots
            intact_model_class: Class for intact fraction model
            dna_model_class: Class for DNA model (optional)
        """
        self.output_dir = output_dir
        self.intact_model_class = intact_model_class
        self.dna_model_class = dna_model_class

        # Will be populated during LOOCV
        self.fold_results = []
        self.fold_metrics_intact = defaultdict(list)
        self.fold_metrics_dna = defaultdict(list)
        self.fold_params = defaultdict(list)
        self.all_loocv_predictions = []

    def run_loocv(self,
                  df_raw: pd.DataFrame,
                  df_intact_fit_base: pd.DataFrame = None,
                  model_variant: str = None) -> Dict[str, Any]:
        """
        Run Leave-One-Out Cross-Validation on the dataset.

        Args:
            df_raw: Full raw dataset
            df_intact_fit_base: Filtered dataset for intact model fitting (optional)
            model_variant: Name of model variant for output path

        Returns:
            Dictionary with LOOCV results summary
        """
        # Create separate output directories for intact and DNA model results
        self.intact_output_dir = self.output_dir / "intact_model"
        self.intact_output_dir.mkdir(parents=True, exist_ok=True)

        if model_variant:
            self.dna_output_dir = self.output_dir / f"dna_model_{model_variant}"
        else:
            self.dna_output_dir = self.output_dir / "dna_model"
        self.dna_output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare base data for LOOCV
        if df_intact_fit_base is None:
            df_intact_fit_base = df_raw.copy()

        df_loocv_base_raw = df_raw.dropna(subset=['experiment_id']).copy()
        df_loocv_base_intact_fit = df_intact_fit_base.dropna(subset=['experiment_id']).copy()

        # --- DEBUG INFORMATION ---
        print("\nDEBUG LOOCV DATA:")
        print(f"  Raw data shape: {df_raw.shape}")
        print(f"  Raw data columns: {df_raw.columns.tolist()}")
        print(f"  Intact fit base shape: {df_intact_fit_base.shape}")
        print(f"  Raw with valid experiment_id shape: {df_loocv_base_raw.shape}")
        print(f"  Intact fit with valid experiment_id shape: {df_loocv_base_intact_fit.shape}")

        # Check experiment_id values
        exp_ids = df_loocv_base_raw['experiment_id'].unique()
        print(f"  Unique experiment_ids: {exp_ids}")

        experiment_ids = sorted(df_loocv_base_raw['experiment_id'].unique())
        n_folds = len(experiment_ids)

        print(f"Starting LOOCV with {n_folds} folds.")
        print(f"  Intact model results will be saved to: {self.intact_output_dir}")
        print(f"  DNA model results will be saved to: {self.dna_output_dir}")

        if n_folds < 2:
            print("LOOCV requires at least 2 experiments.")
            return {"error": "Insufficient experiments for LOOCV"}

        # Reset containers
        self.fold_results = []
        self.fold_metrics_intact = defaultdict(list)
        self.fold_metrics_dna = defaultdict(list)
        self.fold_params = defaultdict(list)
        self.all_loocv_predictions = []
        self.all_intact_predictions = []

        # --- LOOCV Loop ---
        for i, held_out_id in enumerate(experiment_ids):
            print(f"\n--- LOOCV Fold {i + 1}/{n_folds}: Holding out Exp {held_out_id} ---")

            # --- 1. Split Data ---
            train_ids = [eid for eid in experiment_ids if eid != held_out_id]
            df_intact_fit_train = df_loocv_base_intact_fit[
                df_loocv_base_intact_fit['experiment_id'].isin(train_ids)].copy()
            df_raw_train = df_loocv_base_raw[df_loocv_base_raw['experiment_id'].isin(train_ids)].copy()
            df_raw_test = df_loocv_base_raw[df_loocv_base_raw['experiment_id'] == held_out_id].copy()

            # Debug logging for this fold
            print(f"  Train data: {len(df_raw_train)} rows, Test data: {len(df_raw_test)} rows")

            if df_intact_fit_train.empty or df_raw_train.empty or df_raw_test.empty:
                print(f"  Skipping fold {i + 1} due to empty train or test set after splitting.")
                continue

            # --- 2. Train and Evaluate Models ---
            fold_result = self._run_fold(df_intact_fit_train, df_raw_train, df_raw_test)

            # Skip if fold processing failed
            if not fold_result.get("success", False):
                print(f"  Skipping fold {i + 1} due to model fitting/prediction failure.")
                continue

            # --- 3. Store Fold Results ---
            self.fold_results.append(fold_result)
            self.all_loocv_predictions.append(fold_result["test_predictions"])

            # Store metrics
            metrics_intact = fold_result.get("metrics_intact", {})
            metrics_dna = fold_result.get("metrics_dna", {})

            print(f"  Fold Intact Metrics: {metrics_intact}")
            print(f"  Fold DNA Metrics: {metrics_dna}")

            # Append metrics
            for key, value in (metrics_intact or {}).items():
                self.fold_metrics_intact[key].append(value)
            for key, value in (metrics_dna or {}).items():
                self.fold_metrics_dna[key].append(value)

            # Store fitted parameters
            params = fold_result.get("params", {})
            for param_name, param_value in params.items():
                self.fold_params[param_name].append(param_value)

        # Check if we have any successful folds
        if not self.fold_results:
            print("NO SUCCESSFUL FOLDS COMPLETED!")
            return {"error": "No successful LOOCV folds"}

        # --- Generate LOOCV Summary ---
        return self._generate_loocv_summary(model_variant)

    def _run_fold(self,
                  df_intact_fit_train: pd.DataFrame,
                  df_raw_train: pd.DataFrame,
                  df_raw_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a single LOOCV fold.

        Args:
            df_intact_fit_train: Training data for intact model fitting
            df_raw_train: Full training data
            df_raw_test: Test data (held-out experiment)

        Returns:
            Dictionary with fold results
        """
        result = {
            "success": False,
            "params": {},
            "metrics_intact": None,
            "metrics_dna": None,
            "test_predictions": None
        }

        # More detailed debugging
        print("\n  DEBUG FOLD DATA:")
        print(f"    df_intact_fit_train shape: {df_intact_fit_train.shape}")
        if not df_intact_fit_train.empty:
            print(f"    df_intact_fit_train columns: {df_intact_fit_train.columns.tolist()}")
            for col in ['total_passages_650', 'total_passages_1000', 'intact_biomass_percent', 'observed_frac']:
                if col in df_intact_fit_train.columns:
                    nan_count = df_intact_fit_train[col].isna().sum()
                    print(f"    {col} NaN count: {nan_count} ({nan_count / len(df_intact_fit_train) * 100:.1f}%)")

        print(f"    df_raw_train shape: {df_raw_train.shape}")
        print(f"    df_raw_test shape: {df_raw_test.shape}")

        # Check that input data contains required columns
        required_cols = ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step',
                         'total_passages_650', 'total_passages_1000', 'intact_biomass_percent']

        if not all(col in df_intact_fit_train.columns for col in required_cols):
            missing = set(required_cols) - set(df_intact_fit_train.columns)
            print(f"  Error: Missing required columns for intact model training: {missing}")
            return result

        if df_intact_fit_train.empty:
            print("  Error: Empty training data for intact model.")
            return result

        if df_raw_test.empty:
            print("  Error: Empty test data.")
            return result

        try:
            # --- 1. Create and train intact model ---
            intact_model = self.intact_model_class()

            print("  Fitting intact model...")
            intact_success = intact_model.fit(df_intact_fit_train)

            if not intact_success:
                print("  Failed to fit intact model.")
                print("  Inspect fit_intact_model result to see why.")
                return result

            # Store intact model parameters
            for key, value in intact_model.params.items():
                result["params"][key] = value
                print(f"  Intact model parameter: {key} = {value}")

            # --- 2. Generate intact predictions for both train and test data ---
            print("  Generating intact predictions...")
            df_train_intact_pred = intact_model.predict(df_raw_train)
            df_test_intact_pred = intact_model.predict(df_raw_test)

            # Debug intact predictions
            if 'intact_frac_pred' in df_test_intact_pred.columns:
                nan_count = df_test_intact_pred['intact_frac_pred'].isna().sum()
                print(
                    f"  Test intact predictions NaN count: {nan_count} ({nan_count / len(df_test_intact_pred) * 100:.1f}%)")
                if not df_test_intact_pred['intact_frac_pred'].isnull().all():
                    print(
                        f"  Intact prediction range: {df_test_intact_pred['intact_frac_pred'].min():.3f} to {df_test_intact_pred['intact_frac_pred'].max():.3f}")
            else:
                print("  ERROR: 'intact_frac_pred' column missing in test predictions!")

            # Check if predictions are valid
            if 'intact_frac_pred' not in df_test_intact_pred.columns or df_test_intact_pred[
                'intact_frac_pred'].isnull().all():
                print("  Error: Intact model produced no valid predictions for test data.")
                return result

            # --- 3. Evaluate intact model ---
            print("  Evaluating intact model...")
            metrics_intact_test = calculate_metrics(df_test_intact_pred)
            result["metrics_intact"] = metrics_intact_test
            print(f"  Intact metrics: {metrics_intact_test}")

            # If no DNA model class provided, stop here
            if self.dna_model_class is None:
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # --- 4. Create and train DNA model ---
            print("  Fitting DNA model...")
            dna_model = self.dna_model_class()
            dna_success = dna_model.fit(df_train_intact_pred)

            if not dna_success:
                print("  Failed to fit DNA model.")
                # Still return intact results
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # Store DNA model parameters
            for key, value in dna_model.params.items():
                result["params"][key] = value
                print(f"  DNA model parameter: {key} = {value}")

            # --- 5. Generate DNA predictions for test data ---
            print("  Generating DNA predictions...")
            df_test_pred_final = dna_model.predict(df_test_intact_pred)

            # Debug DNA predictions
            if 'dna_pred' in df_test_pred_final.columns:
                nan_count = df_test_pred_final['dna_pred'].isna().sum()
                print(
                    f"  Test DNA predictions NaN count: {nan_count} ({nan_count / len(df_test_pred_final) * 100:.1f}%)")
                if not df_test_pred_final['dna_pred'].isnull().all():
                    print(
                        f"  DNA prediction range: {df_test_pred_final['dna_pred'].min():.3f} to {df_test_pred_final['dna_pred'].max():.3f}")
            else:
                print("  ERROR: 'dna_pred' column missing in test predictions!")

            # Check if predictions are valid
            if 'dna_pred' not in df_test_pred_final.columns or df_test_pred_final['dna_pred'].isnull().all():
                print("  Warning: DNA model produced no valid predictions for test data.")
                # Return results with intact predictions only
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # --- 6. Evaluate DNA model ---
            print("  Evaluating DNA model...")
            metrics_dna_test = calculate_dna_metrics(df_test_pred_final)
            result["metrics_dna"] = metrics_dna_test
            print(f"  DNA metrics: {metrics_dna_test}")

            result["success"] = True
            result["test_predictions"] = df_test_pred_final

            return result

        except Exception as e:
            import traceback
            print(f"  Error during fold execution: {e}")
            traceback.print_exc()
            return result

    def _generate_loocv_summary(self, model_variant: str = None) -> Dict[str, Any]:
        """
        Generate summary of LOOCV results.

        Args:
            model_variant: Name of model variant for output path

        Returns:
            Dictionary with LOOCV summary metrics
        """
        if not self.fold_results:
            print("No LOOCV folds completed successfully.")
            return {"error": "No successful LOOCV folds"}

        try:
            # --- Extract intact and DNA predictions ---
            df_loocv_all_preds = pd.concat(self.all_loocv_predictions, ignore_index=True)

            # Extract intact predictions to save separately
            if 'intact_frac_pred' in df_loocv_all_preds.columns:
                df_intact_preds = df_loocv_all_preds[['experiment_id', 'biomass_type', 'wash_procedure',
                                                      'process_step', 'observed_frac', 'intact_frac_pred']]
                df_intact_preds = df_intact_preds.dropna(subset=['intact_frac_pred'])
            else:
                df_intact_preds = pd.DataFrame()

            # --- 1. Save predictions to appropriate directories ---
            # Save intact predictions
            if not df_intact_preds.empty:
                intact_predictions_path = self.intact_output_dir / "intact_loocv_predictions.csv"
                try:
                    df_intact_preds.to_csv(intact_predictions_path, index=False)
                    print(f"Saved intact LOOCV predictions to {intact_predictions_path}")
                except Exception as e:
                    print(f"Error saving intact LOOCV predictions: {e}")

            # Save DNA predictions
            dna_predictions_path = self.dna_output_dir / "dna_loocv_predictions.csv"
            try:
                df_loocv_all_preds.to_csv(dna_predictions_path, index=False)
                print(f"Saved complete LOOCV predictions to {dna_predictions_path}")
            except Exception as e:
                print(f"Error saving DNA LOOCV predictions: {e}")

            # --- 2. Calculate overall metrics ---
            print("\nOverall LOOCV Metrics (Calculated across all folds' predictions):")
            overall_metrics_intact = calculate_metrics(df_loocv_all_preds)
            overall_metrics_dna = calculate_dna_metrics(df_loocv_all_preds)

            print(f"  Overall Intact: {overall_metrics_intact}")
            print(f"  Overall DNA: {overall_metrics_dna}")

            # --- 3. Calculate average metrics across folds ---
            print("\nAverage Metrics Across Folds:")
            avg_metrics_intact = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
                                  for k, v in self.fold_metrics_intact.items() if v}
            avg_metrics_dna = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
                               for k, v in self.fold_metrics_dna.items() if v}

            print(f"  Avg Fold Intact: {avg_metrics_intact}")
            print(f"  Avg Fold DNA: {avg_metrics_dna}")

            # --- 4. Summarize parameters across folds ---
            print("\nParameter Summary Across Folds (Mean ± Std Dev):")
            param_summary = {}
            intact_params = {}
            dna_params = {}

            for param in self.fold_params.keys():
                values = self.fold_params.get(param, [])
                if not values:
                    param_summary[param] = "N/A"
                    continue

                valid_values = [v for v in values if pd.notna(v)]
                if valid_values:
                    mean_val, std_val = np.mean(valid_values), np.std(valid_values)
                    # Format based on parameter type
                    if param == 'k':
                        fmt = "{:.3e} ± {:.3e}"
                    elif param in ['C_release_fresh', 'C_release_frozen']:
                        fmt = "{:.1f} ± {:.1f}"
                    else:
                        fmt = "{:.3f} ± {:.3f}"
                    param_value = fmt.format(mean_val, std_val)
                    param_summary[param] = param_value

                    # Categorize parameter for separate intact/DNA summaries
                    if param in ['k', 'alpha']:
                        intact_params[param] = param_value
                    else:
                        dna_params[param] = param_value
                else:
                    param_summary[param] = "N/A (No success)"

            for p, v in param_summary.items():
                print(f"  {p}: {v}")

            # --- 5. Generate plots in appropriate directories ---
            print("\nGenerating LOOCV Plots...")
            try:
                # Check if intact predictions exist to plot
                if not df_intact_preds.empty:
                    plot_parity(df_intact_preds, self.intact_output_dir)
                    print(f"  Saved intact parity plot to {self.intact_output_dir}")

                # Check if DNA predictions exist to plot
                if 'dna_pred' in df_loocv_all_preds.columns and not df_loocv_all_preds['dna_pred'].isnull().all():
                    plot_title = f"LOOCV ({model_variant})" if model_variant else "LOOCV"
                    plot_dna_parity_matplotlib(df_loocv_all_preds, self.dna_output_dir, plot_title)
                    print(f"  Saved DNA parity plot to {self.dna_output_dir}")
            except Exception as e:
                print(f"  ERROR during LOOCV plotting: {e}")
                import traceback
                traceback.print_exc()

            # --- 6. Create and save separate summaries for intact and DNA models ---
            # Intact model summary
            intact_summary = {
                "n_folds": len(self.fold_results),
                "overall_metrics": {k: float(v) if isinstance(v, (np.number, np.bool_)) else v
                                    for k, v in (overall_metrics_intact or {}).items()},
                "average_fold_metrics": avg_metrics_intact,
                "parameter_summary": intact_params
            }

            intact_summary_path = self.intact_output_dir / "intact_loocv_summary.json"
            try:
                with open(intact_summary_path, 'w') as f:
                    json.dump(intact_summary, f, indent=4, sort_keys=True)
                print(f"Saved intact model LOOCV summary to {intact_summary_path}")
            except Exception as e:
                print(f"Error saving intact model LOOCV summary: {e}")

            # DNA model summary
            dna_summary = {
                "model_variant": model_variant,
                "n_folds": len(self.fold_results),
                "overall_metrics": {k: float(v) if isinstance(v, (np.number, np.bool_)) else v
                                    for k, v in (overall_metrics_dna or {}).items()},
                "average_fold_metrics": avg_metrics_dna,
                "parameter_summary": dna_params
            }

            dna_summary_path = self.dna_output_dir / "dna_loocv_summary.json"
            try:
                with open(dna_summary_path, 'w') as f:
                    json.dump(dna_summary, f, indent=4, sort_keys=True)
                print(f"Saved DNA model LOOCV summary to {dna_summary_path}")
            except Exception as e:
                print(f"Error saving DNA model LOOCV summary: {e}")

            # Combined summary (for backward compatibility)
            loocv_summary = {
                "model_variant": model_variant,
                "n_folds": len(self.fold_results),
                "overall_metrics_intact": intact_summary["overall_metrics"],
                "overall_metrics_dna": dna_summary["overall_metrics"],
                "average_fold_metrics_intact": avg_metrics_intact,
                "average_fold_metrics_dna": avg_metrics_dna,
                "parameter_summary": param_summary
            }

            return loocv_summary

        except Exception as e:
            import traceback
            print(f"Error generating LOOCV summary: {e}")
            traceback.print_exc()
            return {"error": f"Error generating LOOCV summary: {str(e)}"}