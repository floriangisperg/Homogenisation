# src/analysis/cross_validation.py
import logging
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
from .unified_visualization import VisualizationManager  # Updated import

# Configure module logger
logger = logging.getLogger(__name__)


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

        logger.info("CrossValidator initialized with output_dir: %s", output_dir)
        logger.debug("Using intact_model_class: %s", intact_model_class.__name__)
        if dna_model_class:
            logger.debug("Using dna_model_class: %s", dna_model_class.__name__)
        else:
            logger.debug("No DNA model class provided")

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

        # Create visualization managers for intact and DNA results
        self.intact_viz = VisualizationManager(self.intact_output_dir)
        self.dna_viz = VisualizationManager(self.dna_output_dir)

        # Prepare base data for LOOCV
        if df_intact_fit_base is None:
            df_intact_fit_base = df_raw.copy()
            logger.debug("No intact fit base provided, using raw data")

        df_loocv_base_raw = df_raw.dropna(subset=['experiment_id']).copy()
        df_loocv_base_intact_fit = df_intact_fit_base.dropna(subset=['experiment_id']).copy()

        # Log data shapes and experiment IDs
        logger.debug("Raw data shape: %s", df_raw.shape)
        logger.debug("Intact fit base shape: %s", df_intact_fit_base.shape)
        logger.debug("Raw with valid experiment_id shape: %s", df_loocv_base_raw.shape)
        logger.debug("Intact fit with valid experiment_id shape: %s", df_loocv_base_intact_fit.shape)

        experiment_ids = sorted(df_loocv_base_raw['experiment_id'].unique())
        n_folds = len(experiment_ids)
        logger.info("Starting LOOCV with %d folds (experiment IDs: %s)", n_folds, experiment_ids)
        logger.info("Intact model results will be saved to: %s", self.intact_output_dir)
        logger.info("DNA model results will be saved to: %s", self.dna_output_dir)

        if n_folds < 2:
            logger.error("LOOCV requires at least 2 experiments, only found %d", n_folds)
            return {"error": "Insufficient experiments for LOOCV"}

        # Reset containers
        self.fold_results = []
        self.fold_metrics_intact = defaultdict(list)
        self.fold_metrics_dna = defaultdict(list)
        self.fold_params = defaultdict(list)
        self.all_loocv_predictions = []

        # --- LOOCV Loop ---
        for i, held_out_id in enumerate(experiment_ids):
            logger.info("--- LOOCV Fold %d/%d: Holding out Exp %d ---", i + 1, n_folds, held_out_id)

            # --- 1. Split Data ---
            train_ids = [eid for eid in experiment_ids if eid != held_out_id]
            df_intact_fit_train = df_loocv_base_intact_fit[
                df_loocv_base_intact_fit['experiment_id'].isin(train_ids)].copy()
            df_raw_train = df_loocv_base_raw[df_loocv_base_raw['experiment_id'].isin(train_ids)].copy()
            df_raw_test = df_loocv_base_raw[df_loocv_base_raw['experiment_id'] == held_out_id].copy()

            # Debug logging for this fold
            logger.debug("Train data: %d rows, Test data: %d rows", len(df_raw_train), len(df_raw_test))

            if df_intact_fit_train.empty or df_raw_train.empty or df_raw_test.empty:
                logger.warning("Skipping fold %d due to empty train or test set after splitting.", i + 1)
                continue

            # --- 2. Train and Evaluate Models ---
            fold_result = self._run_fold(df_intact_fit_train, df_raw_train, df_raw_test)

            # Skip if fold processing failed
            if not fold_result.get("success", False):
                logger.warning("Skipping fold %d due to model fitting/prediction failure.", i + 1)
                continue

            # --- 3. Store Fold Results ---
            self.fold_results.append(fold_result)
            self.all_loocv_predictions.append(fold_result["test_predictions"])

            # Store metrics
            metrics_intact = fold_result.get("metrics_intact", {})
            metrics_dna = fold_result.get("metrics_dna", {})

            logger.info("Fold Intact Metrics: %s", metrics_intact)
            logger.info("Fold DNA Metrics: %s", metrics_dna)

            # Append metrics
            for key, value in (metrics_intact or {}).items():
                self.fold_metrics_intact[key].append(value)
            for key, value in (metrics_dna or {}).items():
                self.fold_metrics_dna[key].append(value)

            # Store fitted parameters
            params = fold_result.get("params", {})
            for param_name, param_value in params.items():
                self.fold_params[param_name].append(param_value)

            # --- 4. Generate fold-specific visualizations ---
            test_predictions = fold_result.get("test_predictions")
            if test_predictions is not None and not test_predictions.empty:
                # Generate test prediction plots for this fold
                logger.info("Generating test prediction plots for fold %d", i + 1)
                # subdir = f"fold_{i + 1}"

                # Plot test predictions vs process step
                self.dna_viz.plot_cv_test_predictions(
                    test_predictions,
                    held_out_id,
                )

                # If intact model parameters are available, also generate other plots
                if "k" in params and "alpha" in params:
                    # Generate yield contour plot with fold-specific parameters
                    k, alpha = params.get("k"), params.get("alpha")

                    # Determine F0 based on test data
                    f0 = 1.0  # Default for fresh biomass
                    test_biomass_type = test_predictions["biomass_type"].iloc[0]
                    if test_biomass_type.lower() == "frozen biomass" and "intact_biomass_percent" in test_predictions.columns:
                        f0_val = test_predictions["intact_biomass_percent"].iloc[0] / 100.0
                        if not pd.isna(f0_val) and 0 < f0_val <= 1:
                            f0 = f0_val

                    # Generate fold-specific yield contour plot
                    self.dna_viz.plot_yield_contour(k, alpha, F0=f0,)

        # Check if we have any successful folds
        if not self.fold_results:
            logger.error("NO SUCCESSFUL FOLDS COMPLETED!")
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

        # Log fold data details
        logger.debug("df_intact_fit_train shape: %s", df_intact_fit_train.shape)
        if not df_intact_fit_train.empty:
            # Log NaN counts for key columns
            for col in ['total_passages_650', 'total_passages_1000', 'intact_biomass_percent', 'observed_frac']:
                if col in df_intact_fit_train.columns:
                    nan_count = df_intact_fit_train[col].isna().sum()
                    logger.debug("%s NaN count: %d (%.1f%%)", col, nan_count,
                                 nan_count / len(df_intact_fit_train) * 100)

        logger.debug("df_raw_train shape: %s", df_raw_train.shape)
        logger.debug("df_raw_test shape: %s", df_raw_test.shape)

        # Check that input data contains required columns
        required_cols = ['experiment_id', 'biomass_type', 'wash_procedure', 'process_step',
                         'total_passages_650', 'total_passages_1000', 'intact_biomass_percent']

        if not all(col in df_intact_fit_train.columns for col in required_cols):
            missing = set(required_cols) - set(df_intact_fit_train.columns)
            logger.error("Missing required columns for intact model training: %s", missing)
            return result

        if df_intact_fit_train.empty:
            logger.error("Empty training data for intact model.")
            return result

        if df_raw_test.empty:
            logger.error("Empty test data.")
            return result

        try:
            # --- 1. Create and train intact model ---
            intact_model = self.intact_model_class()

            logger.info("Fitting intact model...")
            intact_success = intact_model.fit(df_intact_fit_train)

            if not intact_success:
                logger.error("Failed to fit intact model.")
                return result

            # Store intact model parameters
            for key, value in intact_model.params.items():
                result["params"][key] = value
                logger.info("Intact model parameter: %s = %s", key, value)

            # --- 2. Generate intact predictions for both train and test data ---
            logger.info("Generating intact predictions...")
            df_train_intact_pred = intact_model.predict(df_raw_train)
            df_test_intact_pred = intact_model.predict(df_raw_test)

            # Debug intact predictions
            if 'intact_frac_pred' in df_test_intact_pred.columns:
                nan_count = df_test_intact_pred['intact_frac_pred'].isna().sum()
                logger.debug("Test intact predictions NaN count: %d (%.1f%%)",
                             nan_count, nan_count / len(df_test_intact_pred) * 100)
                if not df_test_intact_pred['intact_frac_pred'].isnull().all():
                    logger.debug("Intact prediction range: %.3f to %.3f",
                                 df_test_intact_pred['intact_frac_pred'].min(),
                                 df_test_intact_pred['intact_frac_pred'].max())
            else:
                logger.error("'intact_frac_pred' column missing in test predictions!")

            # Check if predictions are valid
            if 'intact_frac_pred' not in df_test_intact_pred.columns or df_test_intact_pred[
                'intact_frac_pred'].isnull().all():
                logger.error("Intact model produced no valid predictions for test data.")
                return result

            # --- 3. Evaluate intact model ---
            logger.info("Evaluating intact model...")
            metrics_intact_test = calculate_metrics(df_test_intact_pred)
            result["metrics_intact"] = metrics_intact_test
            logger.info("Intact metrics: %s", metrics_intact_test)

            # If no DNA model class provided, stop here
            if self.dna_model_class is None:
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # --- 4. Create and train DNA model ---
            logger.info("Fitting DNA model...")
            dna_model = self.dna_model_class()
            dna_success = dna_model.fit(df_train_intact_pred)

            if not dna_success:
                logger.warning("Failed to fit DNA model.")
                # Still return intact results
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # Store DNA model parameters
            for key, value in dna_model.params.items():
                result["params"][key] = value
                logger.info("DNA model parameter: %s = %s", key, value)

            # --- 5. Generate DNA predictions for test data ---
            logger.info("Generating DNA predictions...")
            df_test_pred_final = dna_model.predict(df_test_intact_pred)

            # Debug DNA predictions
            if 'dna_pred' in df_test_pred_final.columns:
                nan_count = df_test_pred_final['dna_pred'].isna().sum()
                logger.debug("Test DNA predictions NaN count: %d (%.1f%%)",
                             nan_count, nan_count / len(df_test_pred_final) * 100)
                if not df_test_pred_final['dna_pred'].isnull().all():
                    logger.debug("DNA prediction range: %.3f to %.3f",
                                 df_test_pred_final['dna_pred'].min(),
                                 df_test_pred_final['dna_pred'].max())
            else:
                logger.error("'dna_pred' column missing in test predictions!")

            # Check if predictions are valid
            if 'dna_pred' not in df_test_pred_final.columns or df_test_pred_final['dna_pred'].isnull().all():
                logger.warning("DNA model produced no valid predictions for test data.")
                # Return results with intact predictions only
                result["success"] = True
                result["test_predictions"] = df_test_intact_pred
                return result

            # --- 6. Evaluate DNA model ---
            logger.info("Evaluating DNA model...")
            metrics_dna_test = calculate_dna_metrics(df_test_pred_final)
            result["metrics_dna"] = metrics_dna_test
            logger.info("DNA metrics: %s", metrics_dna_test)

            result["success"] = True
            result["test_predictions"] = df_test_pred_final

            return result

        except Exception as e:
            import traceback
            logger.exception("Error during fold execution: %s", e)
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
            logger.error("No LOOCV folds completed successfully.")
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
                    logger.info("Saved intact LOOCV predictions to %s", intact_predictions_path)
                except Exception as e:
                    logger.error("Error saving intact LOOCV predictions: %s", e)

            # Save DNA predictions
            dna_predictions_path = self.dna_output_dir / "dna_loocv_predictions.csv"
            try:
                df_loocv_all_preds.to_csv(dna_predictions_path, index=False)
                logger.info("Saved complete LOOCV predictions to %s", dna_predictions_path)
            except Exception as e:
                logger.error("Error saving DNA LOOCV predictions: %s", e)

            # --- 2. Calculate overall metrics ---
            logger.info("Calculating overall LOOCV metrics...")
            overall_metrics_intact = calculate_metrics(df_loocv_all_preds)
            overall_metrics_dna = calculate_dna_metrics(df_loocv_all_preds)

            logger.info("Overall Intact: %s", overall_metrics_intact)
            logger.info("Overall DNA: %s", overall_metrics_dna)

            # --- 3. Calculate average metrics across folds ---
            logger.info("Calculating average metrics across folds...")
            avg_metrics_intact = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
                                  for k, v in self.fold_metrics_intact.items() if v}
            avg_metrics_dna = {k: f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"
                               for k, v in self.fold_metrics_dna.items() if v}

            logger.info("Avg Fold Intact: %s", avg_metrics_intact)
            logger.info("Avg Fold DNA: %s", avg_metrics_dna)

            # --- 4. Summarize parameters across folds ---
            logger.info("Summarizing parameters across folds...")
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
                logger.info("%s: %s", p, v)

            # --- 5. Generate plots in appropriate directories ---
            logger.info("Generating LOOCV plots...")
            try:
                # Check if intact predictions exist to plot
                if not df_intact_preds.empty:
                    # Basic parity plot
                    self.intact_viz.plot_intact_parity(df_intact_preds)
                    logger.info("Saved intact parity plot to %s", self.intact_output_dir)

                    # Generate additional intact plots if we have enough parameters
                    k_values = [v for v in self.fold_params.get('k', []) if pd.notna(v)]
                    alpha_values = [v for v in self.fold_params.get('alpha', []) if pd.notna(v)]

                    if k_values and alpha_values:
                        # Use average parameters from cross-validation
                        mean_k = np.mean(k_values)
                        mean_alpha = np.mean(alpha_values)
                        logger.info("Generating additional intact model plots with mean parameters")

                        # Make sure we have cumulative dose for overview plot
                        if 'cumulative_dose' not in df_intact_preds.columns:
                            # Try to add it using the mean parameters
                            try:
                                from .data_processing import add_cumulative_dose
                                df_intact_preds = add_cumulative_dose(df_intact_preds, mean_k, mean_alpha)
                            except Exception as e:
                                logger.warning("Could not add cumulative_dose for overview plot: %s", e)

                        # Generate overview plot if we can
                        if 'cumulative_dose' in df_intact_preds.columns:
                            self.intact_viz.plot_overview_fitted(df_intact_preds, mean_k, mean_alpha)
                            logger.info("Created overview intact model plot")

                        # Generate yield contour plots for both biomass types
                        self.intact_viz.plot_yield_contour(mean_k, mean_alpha, F0=1.0)  # Fresh biomass
                        logger.info("Created yield contour plot for fresh biomass")

                        # Add frozen biomass contour if applicable
                        if "frozen biomass" in df_intact_preds["biomass_type"].str.lower().unique():
                            # Choose a reasonable F0 for frozen biomass
                            frozen_F0 = 0.8  # Default if we can't find a better value

                            # Try to get a value from data if possible
                            frozen_samples = df_intact_preds[
                                df_intact_preds["biomass_type"].str.lower() == "frozen biomass"]
                            if not frozen_samples.empty and 'intact_biomass_percent' in frozen_samples.columns:
                                first_value = frozen_samples['intact_biomass_percent'].iloc[0] / 100.0
                                if pd.notna(first_value) and 0 < first_value <= 1:
                                    frozen_F0 = first_value

                            self.intact_viz.plot_yield_contour(mean_k, mean_alpha, F0=frozen_F0)
                            logger.info("Created yield contour plot for frozen biomass")

                # Check if DNA predictions exist to plot
                if 'dna_pred' in df_loocv_all_preds.columns and not df_loocv_all_preds['dna_pred'].isnull().all():
                    plot_title = f"LOOCV ({model_variant})" if model_variant else "LOOCV"
                    self.dna_viz.plot_dna_parity(df_loocv_all_preds, title=plot_title)
                    logger.info("Saved DNA parity plot to %s", self.dna_output_dir)

                    # Generate parameter distribution boxplot if we have sufficient data
                    if len(self.fold_params) > 0 and any(len(v) >= 2 for v in self.fold_params.values()):
                        try:
                            self.dna_viz.create_parameter_boxplot(self.fold_params)
                            logger.info("Saved parameter distribution boxplot")
                        except Exception as e:
                            logger.error("Error creating parameter boxplot: %s", e)

                    # Generate overview plot of DNA vs process step if we have sufficient data
                    try:
                        config_name = model_variant or "loocv"
                        self.dna_viz.plot_dna_vs_step(df_loocv_all_preds, config_name=config_name)
                        logger.info("Saved DNA vs process step overview plot")
                    except Exception as e:
                        logger.error("Error creating DNA vs process step plot: %s", e)

                    # Generate residual plots
                    try:
                        self.dna_viz.plot_residuals(
                            df_loocv_all_preds,
                            'dna_conc',
                            'dna_pred',
                            title=f"{model_variant} Residuals" if model_variant else "DNA Residuals",
                            log_scale=True
                        )
                        logger.info("Saved DNA residual plots")
                    except Exception as e:
                        logger.error("Error creating residual plots: %s", e)
            except Exception as e:
                logger.exception("Error during LOOCV plotting: %s", e)

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
                logger.info("Saved intact model LOOCV summary to %s", intact_summary_path)
            except Exception as e:
                logger.error("Error saving intact model LOOCV summary: %s", e)

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
                logger.info("Saved DNA model LOOCV summary to %s", dna_summary_path)
            except Exception as e:
                logger.error("Error saving DNA model LOOCV summary: %s", e)

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
            logger.exception("Error generating LOOCV summary: %s", e)
            return {"error": f"Error generating LOOCV summary: {str(e)}"}