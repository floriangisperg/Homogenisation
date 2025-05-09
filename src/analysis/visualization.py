# src/analysis/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Dict, Any, Tuple, Optional, List, Union

from analysis.data_processing import compute_cumulative_dose_values


class VisualizationManager:
    """
    Unified visualization manager for all plots in the lysis analysis pipeline.
    Consolidates functionality from the original visualization.py and plotting.py.
    """

    def __init__(self, output_dir: Path, style: str = "darkgrid"):
        """
        Initialize the visualization manager.

        Args:
            output_dir: Base directory for saving plots
            style: Seaborn style to use
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seaborn style
        sns.set_style(style)

        # Color schemes
        self.color_schemes = {
            "primary": "royalblue",
            "secondary": "crimson",
            "tertiary": "forestgreen",
            "observed": "blue",
            "predicted": "red",
            "sequential": sns.color_palette("Set2"),
        }

    def _save_plot(self, fig, filename: str, subdir: str = None, dpi: int = 300, tight_layout: bool = True):
        """
        Helper method to save a plot to the output directory.

        Args:
            fig: Matplotlib figure object
            filename: Name of the file to save
            subdir: Optional subdirectory within output_dir
            dpi: Resolution for saving
            tight_layout: Whether to use tight_layout before saving
        """
        save_dir = self.output_dir
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)

        output_path = save_dir / filename

        try:
            if tight_layout:
                fig.tight_layout()
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"  Saved plot to {output_path}")
            return True
        except Exception as e:
            print(f"  Error saving plot: {e}")
            return False

    def plot_intact_parity(
        self, df: pd.DataFrame, subdir: str = None, title: str = None
    ):
        """
        Create parity plot for intact fraction predictions using improved styling.

        Args:
            df: DataFrame with 'observed_frac', 'intact_frac_pred', etc.
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [
            "observed_frac",
            "intact_frac_pred",
            "wash_procedure",
            "biomass_type",
        ]
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for intact parity plot: {missing}")
            return None

        plot_data = df.dropna(subset=required_cols).copy()
        if plot_data.empty:
            print("Warning: No valid data for intact parity plot.")
            return None

        # Set up figure with dark grid style
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # Define markers for biomass types
        markers = {"fresh biomass": "o", "frozen biomass": "s"}

        # Get unique wash procedures and create a color palette
        wash_types = plot_data["wash_procedure"].unique()
        palette = sns.color_palette("Set2", n_colors=len(wash_types))
        color_map = dict(zip(wash_types, palette))

        # Plot each combination of wash and biomass
        legend_elements = []
        for wash in wash_types:
            for biomass in markers.keys():
                subset = plot_data[
                    (plot_data["wash_procedure"] == wash)
                    & (plot_data["biomass_type"] == biomass)
                ]
                if not subset.empty:
                    ax.scatter(
                        subset["observed_frac"],
                        subset["intact_frac_pred"],
                        marker=markers[biomass],
                        color=color_map[wash],
                        edgecolor="k",
                        s=40,
                        alpha=0.9,
                    )

                    # Add to legend
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker=markers[biomass],
                            color="w",
                            markerfacecolor=color_map[wash],
                            markeredgecolor="k",
                            markersize=8,
                            label=f"{wash}, {biomass}",
                        )
                    )

        # Plot y=x line
        lims = [0, 1]
        ax.plot(lims, lims, "k--", lw=1.5)
        legend_elements.append(
            Line2D([0], [0], color="k", linestyle="--", lw=1.5, label="y = x")
        )

        # Labels and formatting
        ax.set_xlabel("Observed Fraction Intact", fontsize=12)
        ax.set_ylabel("Predicted Fraction Intact", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title("Observed vs. Predicted Fraction Intact", fontsize=14)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

        # Add legend
        ax.legend(handles=legend_elements, fontsize=10)

        # Save figure
        filename = "intact_parity_plot.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    # In plot_dna_parity method, add log_scale parameter
    def plot_dna_parity(
        self,
        df: pd.DataFrame,
        subdir: str = None,
        title: str = None,
        log_scale: bool = True,
    ):
        """
        Create parity plot for DNA concentration predictions with improved styling.

        Args:
            df: DataFrame with 'dna_conc', 'dna_pred', etc.
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override
            log_scale: Whether to use log scaling for axes (default: True)

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [
            "dna_conc",
            "dna_pred",
            "wash_procedure",
            "biomass_type",
            "process_step",
        ]
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for DNA parity plot: {missing}")
            return None

        # Filter data - exclude resuspended biomass and handle log scale requirements
        plot_data_base = df.dropna(subset=["dna_conc", "dna_pred"]).copy()
        plot_data_base = plot_data_base[
            plot_data_base["process_step"].str.lower() != "resuspended biomass"
        ]

        if log_scale:
            # For log scale, filter out non-positive values
            plot_data = plot_data_base[
                (plot_data_base["dna_conc"] > 0) & (plot_data_base["dna_pred"] > 0)
            ].copy()
            if plot_data.empty:
                print(
                    f"Warning: No valid positive data points found for DNA log parity plot. Skipping."
                )
                return None
        else:
            # For linear scale, we can keep zeros but not negatives
            plot_data = plot_data_base[
                (plot_data_base["dna_conc"] >= 0) & (plot_data_base["dna_pred"] >= 0)
            ].copy()
            if plot_data.empty:
                print(
                    f"Warning: No valid data points found for DNA linear parity plot. Skipping."
                )
                return None

        # Set up figure with dark grid style
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # Define markers for biomass types
        markers = {"fresh biomass": "o", "frozen biomass": "s"}

        # Get unique wash procedures and create a color palette
        wash_types = plot_data["wash_procedure"].unique()
        palette = sns.color_palette("Set2", n_colors=len(wash_types))
        color_map = dict(zip(wash_types, palette))

        # Plot each combination of wash and biomass
        legend_elements = []
        for wash in wash_types:
            for biomass in markers.keys():
                subset = plot_data[
                    (plot_data["wash_procedure"] == wash)
                    & (plot_data["biomass_type"] == biomass)
                ]
                if not subset.empty:
                    ax.scatter(
                        subset["dna_conc"],
                        subset["dna_pred"],
                        marker=markers[biomass],
                        color=color_map[wash],
                        edgecolor="k",
                        s=40,
                        alpha=0.9,
                    )

                    # Add to legend
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker=markers[biomass],
                            color="w",
                            markerfacecolor=color_map[wash],
                            markeredgecolor="k",
                            markersize=8,
                            label=f"{wash}, {biomass}",
                        )
                    )

        # Set scales and limits
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Plot y=x line for log scale
            min_val = (
                min(plot_data["dna_conc"].min(), plot_data["dna_pred"].min()) * 0.8
            )
            max_val = (
                max(plot_data["dna_conc"].max(), plot_data["dna_pred"].max()) * 1.2
            )
            ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5)

            # Set log-friendly limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
        else:
            # Plot y=x line for linear scale
            max_x = plot_data["dna_conc"].max() * 1.1
            max_y = plot_data["dna_pred"].max() * 1.1
            max_val = max(max_x, max_y)
            ax.plot([0, max_val], [0, max_val], "k--", lw=1.5)

            # Set linear-friendly limits
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)

        # Add y=x line to legend
        legend_elements.append(
            Line2D([0], [0], color="k", linestyle="--", lw=1.5, label="y = x")
        )

        # Labels and formatting
        ax.set_xlabel("Observed DNA Concentration [ng/µL]", fontsize=12)
        ax.set_ylabel("Predicted DNA Concentration [ng/µL]", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title("Observed vs. Predicted DNA Concentration", fontsize=14)

        # Add legend
        ax.legend(handles=legend_elements, fontsize=10)

        # Save figure
        filename = f"dna_parity_plot_{'log' if log_scale else 'linear'}.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    # New method to plot intact fraction vs process step in a grid
    def plot_intact_vs_process_step(
        self,
        df: pd.DataFrame,
        subdir: str = None,
        title: str = None,
        cv_mode: bool = False,
    ):
        """
        Create grid plot showing intact fraction over process steps for all experiments using improved layout.

        Args:
            df: DataFrame with intact fraction data
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override
            cv_mode: Whether in cross-validation mode (highlights test data)

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [
            "experiment_id",
            "process_step",
            "observed_frac",
            "intact_frac_pred",
            "biomass_type",
            "wash_procedure",
        ]

        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for intact vs process step plot: {missing}")
            return None

        df_valid = df.dropna(subset=["experiment_id"]).copy()
        if df_valid.empty:
            print("No valid data with experiment_id for intact vs process step plot.")
            return None

        # Get unique experiment IDs
        exp_ids = sorted(
            [int(x) for x in df_valid["experiment_id"].unique() if pd.notna(x)]
        )
        n_exps = len(exp_ids)

        if n_exps == 0:
            print("No experiments found for intact vs process step plot.")
            return None

        # Set Seaborn style
        sns.set_style("darkgrid")

        # --- Parameters for fixed subplot dimensions (in inches) ---
        subplot_width = 2  # width of each subplot
        subplot_height = 2  # height of each subplot
        h_spacing = 0.5  # horizontal spacing between subplots
        v_spacing = 1.0  # vertical spacing between rows
        left_margin = 1  # left margin of the figure
        right_margin = 1  # right margin
        bottom_margin = 1  # bottom margin
        top_margin = 1.2  # top margin

        # --- Determine layout based on number of experiments ---
        if n_exps <= 3:
            n_top = n_exps
            n_bottom = 0
        elif n_exps <= 4:
            n_top = 2
            n_bottom = 2
        elif n_exps <= 7:
            n_top = 3
            n_bottom = min(4, n_exps - n_top)
        else:  # More than 7 experiments
            n_top = 4
            n_bottom = 4
            print(f"Warning: More than 8 experiments ({n_exps}), only showing first 8.")
            exp_ids = exp_ids[:8]
            n_exps = 8

        has_top_row = n_top > 0
        has_bottom_row = n_bottom > 0

        # Calculate layout dimensions
        if has_top_row:
            total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing
        else:
            total_top_width = 0

        if has_bottom_row:
            total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
        else:
            total_bottom_width = 0

        # Figure width is the maximum row width plus margins
        fig_width = (
            max(total_top_width, total_bottom_width) + left_margin + right_margin
        )

        # Figure height depends on how many rows we have
        if has_top_row and has_bottom_row:
            fig_height = (
                top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
            )
        elif has_top_row or has_bottom_row:
            fig_height = top_margin + subplot_height + bottom_margin
        else:
            fig_height = (
                top_margin + bottom_margin
            )  # Should never happen as we always have at least 1 experiment

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

        # --- Create axes for each row ---
        axes_list = []

        if has_top_row:
            top_y = (
                (bottom_margin + subplot_height + v_spacing) / fig_height
                if has_bottom_row
                else bottom_margin / fig_height
            )
            top_h = subplot_height / fig_height
            top_left_offset = (
                left_margin
                + (fig_width - left_margin - right_margin - total_top_width) / 2
            )

            for i in range(n_top):
                ax_left = (
                    top_left_offset + i * (subplot_width + h_spacing)
                ) / fig_width
                ax = fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h])
                axes_list.append(ax)

        if has_bottom_row:
            bottom_y = bottom_margin / fig_height
            bottom_h = subplot_height / fig_height
            bottom_left_offset = (
                left_margin
                + (fig_width - left_margin - right_margin - total_bottom_width) / 2
            )

            for i in range(n_bottom):
                ax_left = (
                    bottom_left_offset + i * (subplot_width + h_spacing)
                ) / fig_width
                ax = fig.add_axes(
                    [ax_left, bottom_y, subplot_width / fig_width, bottom_h]
                )
                axes_list.append(ax)

        # Divide experiment IDs between rows
        top_experiments = exp_ids[:n_top]
        bottom_experiments = exp_ids[n_top : n_top + n_bottom]

        # Define standard process step order
        step_order = {
            "resuspended biomass": 0,
            "initial lysis": 1,
            "1st wash": 2,
            "2nd wash": 3,
            "3rd wash": 4,
            "4th wash": 5,
        }

        # Set colors
        observed_color = self.color_schemes["primary"]
        predicted_color = self.color_schemes["secondary"]

        # Plot each experiment
        for i, (ax, exp_id) in enumerate(zip(axes_list, exp_ids)):
            exp_data = df_valid[df_valid["experiment_id"] == exp_id].copy()

            # Check if we have any data
            if exp_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for Exp {exp_id}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.set_title(f"Experiment {exp_id}")
                continue

            # Sort by process step
            exp_data["step_order"] = exp_data["process_step"].map(
                lambda x: step_order.get(str(x).lower(), 999)
            )
            exp_data = exp_data.sort_values("step_order")

            # Get process steps for x-axis
            x_steps = exp_data["process_step"].values

            # Get biomass type and wash procedure for annotation
            biomass = exp_data["biomass_type"].iloc[0]
            wash = exp_data["wash_procedure"].iloc[0]

            # For CV mode, highlight test data if available
            if cv_mode and "is_test" in exp_data.columns:
                # Plot training data with lower alpha
                train_data = exp_data[~exp_data["is_test"]].copy()
                if not train_data.empty:
                    valid_obs = train_data["observed_frac"].notna()
                    if valid_obs.any():
                        ax.plot(
                            train_data["process_step"].values,
                            train_data.loc[valid_obs, "observed_frac"],
                            "o--",
                            color=observed_color,
                            alpha=0.3,
                            markersize=6,
                        )

                    valid_pred = train_data["intact_frac_pred"].notna()
                    if valid_pred.any():
                        ax.plot(
                            train_data["process_step"].values,
                            train_data.loc[valid_pred, "intact_frac_pred"],
                            "x--",
                            color=predicted_color,
                            alpha=0.3,
                            markersize=6,
                        )

                # Plot test data with full alpha
                test_data = exp_data[exp_data["is_test"]].copy()
                if not test_data.empty:
                    valid_obs = test_data["observed_frac"].notna()
                    if valid_obs.any():
                        ax.plot(
                            test_data["process_step"].values,
                            test_data.loc[valid_obs, "observed_frac"],
                            "o-",
                            color=observed_color,
                            linewidth=2,
                            markersize=8,
                        )

                    valid_pred = test_data["intact_frac_pred"].notna()
                    if valid_pred.any():
                        ax.plot(
                            test_data["process_step"].values,
                            test_data.loc[valid_pred, "intact_frac_pred"],
                            "x--",
                            color=predicted_color,
                            linewidth=2,
                            markersize=8,
                        )
            else:
                # Standard plot (not CV)
                valid_obs = exp_data["observed_frac"].notna()
                if valid_obs.any():
                    ax.plot(
                        exp_data["process_step"].values,
                        exp_data.loc[valid_obs, "observed_frac"],
                        "o-",
                        color=observed_color,
                        linewidth=2,
                        markersize=8,
                    )

                valid_pred = exp_data["intact_frac_pred"].notna()
                if valid_pred.any():
                    ax.plot(
                        exp_data["process_step"].values,
                        exp_data.loc[valid_pred, "intact_frac_pred"],
                        "x--",
                        color=predicted_color,
                        linewidth=2,
                        markersize=8,
                    )

            # Formatting
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.set_ylim(-0.05, 1.05)

            # Add process step information
            ax.text(
                0.95,
                0.95,
                f"{wash}\n{biomass}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
            )

            ax.set_title(f"Experiment {exp_id}", fontsize=10)

            # Only add y-label for leftmost plots
            if (has_top_row and i == 0) or (not has_top_row and i == 0):
                ax.set_ylabel("Fraction Intact", fontsize=10)
            else:
                ax.set_ylabel("")

            # Only add x-label for bottom row
            if (has_bottom_row and i >= n_top) or (not has_bottom_row):
                ax.set_xlabel("Process Step", fontsize=10)
            else:
                ax.set_xlabel("")

        # Common legend
        observed_handle = Line2D(
            [],
            [],
            marker="o",
            color=observed_color,
            linestyle="-",
            markersize=6,
            label="Observed",
        )
        predicted_handle = Line2D(
            [],
            [],
            marker="x",
            color=predicted_color,
            linestyle="--",
            markersize=6,
            label="Predicted",
        )

        fig.legend(
            handles=[observed_handle, predicted_handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - (top_margin / 3) / fig_height),
            ncol=2,
            fontsize=10,
        )

        # Add global title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)

        # Save figure
        filename = "intact_vs_process_steps_grid.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    # New method to plot DNA vs process step with option for log or linear scale
    def plot_dna_vs_process_step_grid(
        self,
        df: pd.DataFrame,
        log_scale: bool = True,
        subdir: str = None,
        title: str = None,
        cv_mode: bool = False,
    ):
        """
        Create grid plot showing DNA concentration over process steps for all experiments.

        Args:
            df: DataFrame with DNA concentration data
            log_scale: Whether to use log scale for y-axis (default: True)
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override
            cv_mode: Whether in cross-validation mode (highlights test data)

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [
            "experiment_id",
            "process_step",
            "dna_conc",
            "dna_pred",
            "biomass_type",
            "wash_procedure",
        ]

        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for DNA vs process step plot: {missing}")
            return None

        df_valid = df.dropna(subset=["experiment_id"]).copy()
        if df_valid.empty:
            print("No valid data with experiment_id for DNA vs process step plot.")
            return None

        # Get unique experiment IDs
        exp_ids = sorted(
            [int(x) for x in df_valid["experiment_id"].unique() if pd.notna(x)]
        )
        n_exps = len(exp_ids)

        if n_exps == 0:
            print("No experiments found for DNA vs process step plot.")
            return None

        # Create grid layout
        n_cols = min(3, n_exps)
        n_rows = (n_exps + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        # Define standard process step order
        step_order = {
            "resuspended biomass": 0,
            "initial lysis": 1,
            "1st wash": 2,
            "2nd wash": 3,
            "3rd wash": 4,
            "4th wash": 5,
        }

        # Plot each experiment
        for i, exp_id in enumerate(exp_ids):
            if i >= len(axes):
                break

            ax = axes[i]
            exp_data = df_valid[df_valid["experiment_id"] == exp_id].copy()

            # Check if we have any data
            if exp_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for Exp {exp_id}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax.set_title(f"Experiment {exp_id}")
                continue

            # Sort by process step
            exp_data["step_order"] = exp_data["process_step"].map(
                lambda x: step_order.get(str(x).lower(), 999)
            )
            exp_data = exp_data.sort_values("step_order")

            # Get biomass type and wash procedure for title
            biomass = exp_data["biomass_type"].iloc[0]
            wash = exp_data["wash_procedure"].iloc[0]

            # For CV mode, highlight test data if available
            if cv_mode and "is_test" in exp_data.columns:
                # Filter data based on log scale if needed
                if log_scale:
                    train_data = exp_data[
                        (~exp_data["is_test"])
                        & (exp_data["dna_conc"] > 0)
                        & (exp_data["dna_pred"] > 0)
                    ].copy()
                    test_data = exp_data[
                        (exp_data["is_test"])
                        & (exp_data["dna_conc"] > 0)
                        & (exp_data["dna_pred"] > 0)
                    ].copy()
                else:
                    train_data = exp_data[
                        (~exp_data["is_test"])
                        & (exp_data["dna_conc"] >= 0)
                        & (exp_data["dna_pred"] >= 0)
                    ].copy()
                    test_data = exp_data[
                        (exp_data["is_test"])
                        & (exp_data["dna_conc"] >= 0)
                        & (exp_data["dna_pred"] >= 0)
                    ].copy()

                # Plot training data with lower alpha
                if not train_data.empty:
                    valid_obs = train_data["dna_conc"].notna()
                    if valid_obs.any():
                        ax.plot(
                            train_data.loc[valid_obs, "process_step"],
                            train_data.loc[valid_obs, "dna_conc"],
                            "o--",
                            color="blue",
                            alpha=0.3,
                            markersize=6,
                            label="Train (Observed)",
                        )

                    valid_pred = train_data["dna_pred"].notna()
                    if valid_pred.any():
                        ax.plot(
                            train_data.loc[valid_pred, "process_step"],
                            train_data.loc[valid_pred, "dna_pred"],
                            "x--",
                            color="red",
                            alpha=0.3,
                            markersize=6,
                            label="Train (Predicted)",
                        )

                # Plot test data with full alpha
                if not test_data.empty:
                    valid_obs = test_data["dna_conc"].notna()
                    if valid_obs.any():
                        ax.plot(
                            test_data.loc[valid_obs, "process_step"],
                            test_data.loc[valid_obs, "dna_conc"],
                            "o-",
                            color="blue",
                            linewidth=2,
                            markersize=8,
                            label="Test (Observed)",
                        )

                    valid_pred = test_data["dna_pred"].notna()
                    if valid_pred.any():
                        ax.plot(
                            test_data.loc[valid_pred, "process_step"],
                            test_data.loc[valid_pred, "dna_pred"],
                            "x--",
                            color="red",
                            linewidth=2,
                            markersize=8,
                            label="Test (Predicted)",
                        )
            else:
                # Standard plot (not CV)
                if log_scale:
                    valid_data = exp_data[
                        (exp_data["dna_conc"] > 0) | (exp_data["dna_pred"] > 0)
                    ].copy()
                else:
                    valid_data = exp_data[
                        (exp_data["dna_conc"] >= 0) | (exp_data["dna_pred"] >= 0)
                    ].copy()

                if not valid_data.empty:
                    valid_obs = valid_data["dna_conc"].notna()
                    if valid_obs.any():
                        ax.plot(
                            valid_data.loc[valid_obs, "process_step"],
                            valid_data.loc[valid_obs, "dna_conc"],
                            "o-",
                            color="blue",
                            linewidth=2,
                            markersize=8,
                            label="Observed",
                        )

                    valid_pred = valid_data["dna_pred"].notna()
                    if valid_pred.any():
                        ax.plot(
                            valid_data.loc[valid_pred, "process_step"],
                            valid_data.loc[valid_pred, "dna_pred"],
                            "x--",
                            color="red",
                            linewidth=2,
                            markersize=8,
                            label="Predicted",
                        )

            # Set log scale if requested
            if log_scale:
                ax.set_yscale("log")

            # Formatting
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.set_title(f"Exp {exp_id}: {biomass}, {wash}")
            ax.grid(True, alpha=0.3)

            # Only add x-label for bottom row
            if i >= n_exps - n_cols:
                ax.set_xlabel("Process Step")

            # Only add y-label for leftmost column
            if i % n_cols == 0:
                scale_text = " (Log Scale)" if log_scale else ""
                ax.set_ylabel(f"DNA Conc. [ng/µL]{scale_text}")

            # Add legend only for first plot to avoid clutter
            if i == 0:
                ax.legend(fontsize=8)

        # Hide unused axes
        for i in range(n_exps, len(axes)):
            axes[i].set_visible(False)

        # Add global title
        scale_text = " (Log Scale)" if log_scale else " (Linear Scale)"
        if title:
            plt.suptitle(f"{title}{scale_text}", fontsize=16, y=1.02)
        else:
            plt.suptitle(
                f"DNA Concentration vs Process Steps{scale_text}", fontsize=16, y=1.02
            )

        plt.tight_layout()

        # Save figure
        scale_suffix = "log" if log_scale else "linear"
        filename = f"dna_vs_process_steps_grid_{scale_suffix}.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    # Migrated from plotting.py
    def plot_overview_fitted(self, df_fit: pd.DataFrame, k: float, alpha: float, subdir: str = None):
        """
        Generate overview plot showing both observed and fitted intact fractions vs dose.

        Args:
            df_fit: DataFrame with intact fraction data
            k: Fitted k parameter
            alpha: Fitted alpha parameter
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['experiment_id', 'observed_frac', 'cumulative_dose', 'intact_biomass_percent',
                         'biomass_type', 'wash_procedure']
        if not all(col in df_fit.columns for col in required_cols):
            missing = set(required_cols) - set(df_fit.columns)
            print(f"Missing columns for overview fitted plot: {missing}")
            return None

        df_plot_data = df_fit.dropna(subset=['experiment_id']).copy()
        if df_plot_data.empty:
            print("Warning: No experiments after dropping NA experiment_id.")
            return None

        experiment_ids = sorted(df_plot_data['experiment_id'].unique())
        n_total = len(experiment_ids)
        if n_total == 0:
            print("Warning: No experiments found for overview plot.")
            return None

        # Prepare layout parameters
        n_top = 3 if n_total >= 3 else n_total
        n_bottom = min(4, n_total - n_top)
        has_top_row = n_top > 0
        has_bottom_row = n_bottom > 0

        # Calculate figure dimensions
        subplot_width = 2
        subplot_height = 2
        h_spacing = 0.5
        v_spacing = 1.0 if (has_top_row and has_bottom_row) else 0
        left_margin = 1
        right_margin = 1
        bottom_margin = 1
        top_margin = 1.2

        total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing if has_top_row else 0
        total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing if has_bottom_row else 0

        fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
        if has_top_row and has_bottom_row:
            fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
        elif has_top_row or has_bottom_row:
            fig_height = top_margin + subplot_height + bottom_margin
        else:
            fig_height = top_margin + bottom_margin

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create axes
        axes_list = []
        if has_top_row:
            top_y = (
                                bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_margin / fig_height
            top_h = subplot_height / fig_height
            top_row_content_width = fig_width - left_margin - right_margin
            top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2

            for i in range(n_top):
                ax_left_norm = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width
                ax_width_norm = subplot_width / fig_width
                axes_list.append(fig.add_axes([ax_left_norm, top_y, ax_width_norm, top_h]))

        if has_bottom_row:
            bottom_y = bottom_margin / fig_height
            bottom_h = subplot_height / fig_height
            bottom_row_content_width = fig_width - left_margin - right_margin
            bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2

            for i in range(n_bottom):
                ax_left_norm = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width
                ax_width_norm = subplot_width / fig_width
                axes_list.append(fig.add_axes([ax_left_norm, bottom_y, ax_width_norm, bottom_h]))

        # Define plot colors and shared axis limits
        observed_color = self.color_schemes['primary']
        predicted_color = self.color_schemes['secondary']
        common_yticks = np.arange(0, 1.1, 0.25)
        max_overall_dose = df_plot_data['cumulative_dose'].max()
        xlim_max = max(2, np.ceil(max_overall_dose)) if pd.notna(max_overall_dose) else 2
        common_xlim = (-0.05 * xlim_max, xlim_max * 1.05)

        # Plot each experiment
        for i, (ax, exp_id) in enumerate(zip(axes_list, experiment_ids)):
            exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
            if exp_data.empty:
                continue

            # Plot observed data points
            ax.scatter(
                exp_data['cumulative_dose'],
                exp_data['observed_frac'],
                color=observed_color,
                s=25,
                zorder=5,
                edgecolor='k',
                linewidth=0.5
            )

            # Plot predicted curve
            row0 = exp_data.iloc[0]
            if row0['biomass_type'] == "fresh biomass":
                F0 = 1.0
            else:
                F0 = row0['intact_biomass_percent'] / 100.0
                if pd.isna(F0) or not (0 < F0 <= 1):
                    F0 = row0['observed_frac']
                if pd.isna(F0) or not (0 < F0 <= 1):
                    F0 = None

            if F0 is not None:
                max_dose_exp = exp_data['cumulative_dose'].max()
                if pd.isna(max_dose_exp) or max_dose_exp <= 0:
                    max_dose_exp = 1e-6
                dose_fine = np.linspace(0, max_dose_exp, 200)
                predicted_fine = F0 * np.exp(-dose_fine)
                ax.plot(dose_fine, predicted_fine, color=predicted_color, lw=1.5, zorder=3)

            # Set axis limits and ticks
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(common_yticks)
            ax.set_xlim(common_xlim)

            plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=9)
            plt.setp(ax.get_yticklabels(), fontsize=9)

            # Set labels based on position
            is_leftmost_top = has_top_row and (i == 0)
            is_leftmost_bottom = has_bottom_row and (i == n_top)

            if is_leftmost_top or is_leftmost_bottom:
                ax.set_ylabel("Fraction Intact", fontsize=11)
            else:
                ax.set_ylabel('')
                plt.setp(ax.get_yticklabels(), visible=False)

            is_on_bottom_row = has_bottom_row and (i >= n_top)
            is_only_row = not has_bottom_row

            if is_on_bottom_row or is_only_row:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), visible=False)

            # Add experiment info and title
            wash = row0['wash_procedure']
            biomass = row0['biomass_type']
            ax.text(
                0.95, 0.95,
                f"{wash}\n{biomass}",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
            )

            ax.set_title(f'Exp {exp_id}', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)

        # Add common x-axis label
        if has_top_row or has_bottom_row:
            fig.text(
                0.5, (bottom_margin / 2) / fig_height,
                r"Cumulative Dose ($k \cdot P^{\alpha} \cdot N$)",
                ha='center',
                va='center',
                fontsize=12
            )

        # Add legend
        observed_handle = Line2D(
            [], [],
            marker='o',
            color=observed_color,
            linestyle='None',
            markersize=6,
            markeredgecolor='k',
            markeredgewidth=0.5,
            label='Observed'
        )

        predicted_handle = Line2D(
            [], [],
            color=predicted_color,
            lw=1.5,
            label='Model Fit'
        )

        fig.legend(
            handles=[observed_handle, predicted_handle],
            loc='upper center',
            bbox_to_anchor=(0.5, 1 - (top_margin / fig_height) * 0.2),
            ncol=2,
            fontsize=11
        )

        # Save figure
        filename = "overview_fitted_vs_dose.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_combined_intact_dna_vs_process(
        self,
        df: pd.DataFrame,
        log_scale_dna: bool = True,
        subdir: str = None,
        title: str = None,
        cv_mode: bool = False,
    ):
        """
        Create grid plot showing both intact fraction and DNA concentration vs process steps with dual y-axes.

        Args:
            df: DataFrame with both intact fraction and DNA data
            log_scale_dna: Whether to use log scale for DNA y-axis
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override
            cv_mode: Whether in cross-validation mode (highlights test data)

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [
            "experiment_id",
            "process_step",
            "observed_frac",
            "intact_frac_pred",
            "dna_conc",
            "dna_pred",
            "biomass_type",
            "wash_procedure",
        ]

        # Check if we have all required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing columns for combined plot: {missing}")
            return None

        # Filter to valid data
        df_valid = df.dropna(subset=["experiment_id"]).copy()
        if df_valid.empty:
            print("No valid data with experiment_id for combined plot.")
            return None

        # Get unique experiment IDs
        exp_ids = sorted(
            [int(x) for x in df_valid["experiment_id"].unique() if pd.notna(x)]
        )
        n_exps = len(exp_ids)

        if n_exps == 0:
            print("No experiments found for combined plot.")
            return None

        # Set Seaborn style
        sns.set_style("darkgrid")

        # --- Parameters for fixed subplot dimensions (in inches) ---
        subplot_width = 2.5  # width of each subplot (slightly wider for dual y-axis)
        subplot_height = 2.5  # height of each subplot (slightly taller for dual y-axis)
        h_spacing = 0.7  # horizontal spacing between subplots (wider for dual y-axis)
        v_spacing = 1.2  # vertical spacing between rows
        left_margin = 1.2  # left margin of the figure (wider for dual y-axis)
        right_margin = 1.2  # right margin (wider for dual y-axis)
        bottom_margin = 1.2  # bottom margin
        top_margin = 1.5  # top margin

        # --- Determine layout based on number of experiments ---
        if n_exps <= 3:
            n_top = n_exps
            n_bottom = 0
        elif n_exps <= 4:
            n_top = 2
            n_bottom = 2
        elif n_exps <= 7:
            n_top = 3
            n_bottom = min(4, n_exps - n_top)
        else:  # More than 7 experiments
            n_top = 4
            n_bottom = 4
            print(f"Warning: More than 8 experiments ({n_exps}), only showing first 8.")
            exp_ids = exp_ids[:8]
            n_exps = 8

        has_top_row = n_top > 0
        has_bottom_row = n_bottom > 0

        # Calculate layout dimensions
        if has_top_row:
            total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing
        else:
            total_top_width = 0

        if has_bottom_row:
            total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
        else:
            total_bottom_width = 0

        # Figure width is the maximum row width plus margins
        fig_width = (
            max(total_top_width, total_bottom_width) + left_margin + right_margin
        )

        # Figure height depends on how many rows we have
        if has_top_row and has_bottom_row:
            fig_height = (
                top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
            )
        elif has_top_row or has_bottom_row:
            fig_height = top_margin + subplot_height + bottom_margin
        else:
            fig_height = (
                top_margin + bottom_margin
            )  # Should never happen as we always have at least 1 experiment

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

        # --- Create axes for each row ---
        # For each experiment, we'll create a main ax and a twin ax for dual y-axis
        main_axes = []
        twin_axes = []

        if has_top_row:
            top_y = (
                (bottom_margin + subplot_height + v_spacing) / fig_height
                if has_bottom_row
                else bottom_margin / fig_height
            )
            top_h = subplot_height / fig_height
            top_left_offset = (
                left_margin
                + (fig_width - left_margin - right_margin - total_top_width) / 2
            )

            for i in range(n_top):
                ax_left = (
                    top_left_offset + i * (subplot_width + h_spacing)
                ) / fig_width
                main_ax = fig.add_axes(
                    [ax_left, top_y, subplot_width / fig_width, top_h]
                )
                twin_ax = main_ax.twinx()
                main_axes.append(main_ax)
                twin_axes.append(twin_ax)

        if has_bottom_row:
            bottom_y = bottom_margin / fig_height
            bottom_h = subplot_height / fig_height
            bottom_left_offset = (
                left_margin
                + (fig_width - left_margin - right_margin - total_bottom_width) / 2
            )

            for i in range(n_bottom):
                ax_left = (
                    bottom_left_offset + i * (subplot_width + h_spacing)
                ) / fig_width
                main_ax = fig.add_axes(
                    [ax_left, bottom_y, subplot_width / fig_width, bottom_h]
                )
                twin_ax = main_ax.twinx()
                main_axes.append(main_ax)
                twin_axes.append(twin_ax)

        # Divide experiment IDs between rows
        top_experiments = exp_ids[:n_top]
        bottom_experiments = exp_ids[n_top : n_top + n_bottom]

        # Define standard process step order
        step_order = {
            "resuspended biomass": 0,
            "initial lysis": 1,
            "1st wash": 2,
            "2nd wash": 3,
            "3rd wash": 4,
            "4th wash": 5,
        }

        # Set colors
        intact_obs_color = self.color_schemes["primary"]
        intact_pred_color = self.color_schemes["secondary"]
        dna_obs_color = "green"
        dna_pred_color = "orange"

        # Plot each experiment
        for i, (main_ax, twin_ax, exp_id) in enumerate(
            zip(main_axes, twin_axes, exp_ids)
        ):
            exp_data = df_valid[df_valid["experiment_id"] == exp_id].copy()

            # Check if we have any data
            if exp_data.empty:
                main_ax.text(
                    0.5,
                    0.5,
                    f"No data for Exp {exp_id}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                main_ax.set_title(f"Experiment {exp_id}")
                continue

            # Sort by process step
            exp_data["step_order"] = exp_data["process_step"].map(
                lambda x: step_order.get(str(x).lower(), 999)
            )
            exp_data = exp_data.sort_values("step_order")

            # Get process steps for x-axis
            x_steps = exp_data["process_step"].values

            # Get biomass type and wash procedure for annotation
            biomass = exp_data["biomass_type"].iloc[0]
            wash = exp_data["wash_procedure"].iloc[0]

            # Check if we have any DNA data
            has_dna_data = (
                not exp_data["dna_conc"].isnull().all()
                and not exp_data["dna_pred"].isnull().all()
            )

            # Plot intact fraction on the main axis
            if cv_mode and "is_test" in exp_data.columns:
                # CV mode - highlight test data
                train_data = exp_data[~exp_data["is_test"]].copy()
                test_data = exp_data[exp_data["is_test"]].copy()

                # Plot training data with lower alpha
                if not train_data.empty:
                    valid_obs = train_data["observed_frac"].notna()
                    if valid_obs.any():
                        main_ax.plot(
                            train_data["process_step"].values,
                            train_data.loc[valid_obs, "observed_frac"],
                            "o--",
                            color=intact_obs_color,
                            alpha=0.3,
                            markersize=6,
                        )

                    valid_pred = train_data["intact_frac_pred"].notna()
                    if valid_pred.any():
                        main_ax.plot(
                            train_data["process_step"].values,
                            train_data.loc[valid_pred, "intact_frac_pred"],
                            "x--",
                            color=intact_pred_color,
                            alpha=0.3,
                            markersize=6,
                        )

                # Plot test data with full alpha
                if not test_data.empty:
                    valid_obs = test_data["observed_frac"].notna()
                    if valid_obs.any():
                        main_ax.plot(
                            test_data["process_step"].values,
                            test_data.loc[valid_obs, "observed_frac"],
                            "o-",
                            color=intact_obs_color,
                            linewidth=2,
                            markersize=8,
                        )

                    valid_pred = test_data["intact_frac_pred"].notna()
                    if valid_pred.any():
                        main_ax.plot(
                            test_data["process_step"].values,
                            test_data.loc[valid_pred, "intact_frac_pred"],
                            "x--",
                            color=intact_pred_color,
                            linewidth=2,
                            markersize=8,
                        )

                # Plot DNA data on the twin axis if available
                if has_dna_data:
                    # Filter by log scale requirements if needed
                    if log_scale_dna:
                        train_data_dna = train_data[
                            (train_data["dna_conc"] > 0) | (train_data["dna_pred"] > 0)
                        ].copy()
                        test_data_dna = test_data[
                            (test_data["dna_conc"] > 0) | (test_data["dna_pred"] > 0)
                        ].copy()
                    else:
                        train_data_dna = train_data[
                            (train_data["dna_conc"] >= 0)
                            | (train_data["dna_pred"] >= 0)
                        ].copy()
                        test_data_dna = test_data[
                            (test_data["dna_conc"] >= 0) | (test_data["dna_pred"] >= 0)
                        ].copy()

                    # Plot training DNA data
                    if not train_data_dna.empty:
                        valid_obs = train_data_dna["dna_conc"].notna()
                        if valid_obs.any():
                            twin_ax.plot(
                                train_data_dna["process_step"].values,
                                train_data_dna.loc[valid_obs, "dna_conc"],
                                "s--",
                                color=dna_obs_color,
                                alpha=0.3,
                                markersize=6,
                            )

                        valid_pred = train_data_dna["dna_pred"].notna()
                        if valid_pred.any():
                            twin_ax.plot(
                                train_data_dna["process_step"].values,
                                train_data_dna.loc[valid_pred, "dna_pred"],
                                "d--",
                                color=dna_pred_color,
                                alpha=0.3,
                                markersize=6,
                            )

                    # Plot test DNA data
                    if not test_data_dna.empty:
                        valid_obs = test_data_dna["dna_conc"].notna()
                        if valid_obs.any():
                            twin_ax.plot(
                                test_data_dna["process_step"].values,
                                test_data_dna.loc[valid_obs, "dna_conc"],
                                "s-",
                                color=dna_obs_color,
                                linewidth=2,
                                markersize=8,
                            )

                        valid_pred = test_data_dna["dna_pred"].notna()
                        if valid_pred.any():
                            twin_ax.plot(
                                test_data_dna["process_step"].values,
                                test_data_dna.loc[valid_pred, "dna_pred"],
                                "d--",
                                color=dna_pred_color,
                                linewidth=2,
                                markersize=8,
                            )
            else:
                # Standard mode (not CV)
                # Plot intact fraction data
                valid_obs = exp_data["observed_frac"].notna()
                if valid_obs.any():
                    main_ax.plot(
                        exp_data["process_step"].values,
                        exp_data.loc[valid_obs, "observed_frac"],
                        "o-",
                        color=intact_obs_color,
                        linewidth=2,
                        markersize=8,
                    )

                valid_pred = exp_data["intact_frac_pred"].notna()
                if valid_pred.any():
                    main_ax.plot(
                        exp_data["process_step"].values,
                        exp_data.loc[valid_pred, "intact_frac_pred"],
                        "x--",
                        color=intact_pred_color,
                        linewidth=2,
                        markersize=8,
                    )

                # Plot DNA data on the twin axis if available
                if has_dna_data:
                    # Filter by log scale requirements if needed
                    if log_scale_dna:
                        exp_data_dna = exp_data[
                            (exp_data["dna_conc"] > 0) | (exp_data["dna_pred"] > 0)
                        ].copy()
                    else:
                        exp_data_dna = exp_data[
                            (exp_data["dna_conc"] >= 0) | (exp_data["dna_pred"] >= 0)
                        ].copy()

                    if not exp_data_dna.empty:
                        valid_obs = exp_data_dna["dna_conc"].notna()
                        if valid_obs.any():
                            twin_ax.plot(
                                exp_data_dna["process_step"].values,
                                exp_data_dna.loc[valid_obs, "dna_conc"],
                                "s-",
                                color=dna_obs_color,
                                linewidth=2,
                                markersize=8,
                            )

                        valid_pred = exp_data_dna["dna_pred"].notna()
                        if valid_pred.any():
                            twin_ax.plot(
                                exp_data_dna["process_step"].values,
                                exp_data_dna.loc[valid_pred, "dna_pred"],
                                "d--",
                                color=dna_pred_color,
                                linewidth=2,
                                markersize=8,
                            )

            # Set y-axis limits for intact fraction
            main_ax.set_ylim(-0.05, 1.05)

            # Set y-axis scale for DNA concentration
            if has_dna_data and log_scale_dna:
                twin_ax.set_yscale("log")

            # Format x-axis
            plt.setp(main_ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

            # Add process step annotation
            main_ax.text(
                0.95,
                0.95,
                f"{wash}\n{biomass}",
                transform=main_ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
            )

            # Add title
            main_ax.set_title(f"Experiment {exp_id}", fontsize=10)

            # Only add y-labels for leftmost and rightmost plots
            if (has_top_row and i == 0) or (not has_top_row and i == 0):
                main_ax.set_ylabel(
                    "Fraction Intact", fontsize=10, color=intact_obs_color
                )
            else:
                main_ax.set_ylabel("")

            if has_dna_data:
                if (has_top_row and i == n_top - 1) or (
                    not has_top_row and i == n_exps - 1
                ):
                    twin_ax.set_ylabel(
                        "DNA Conc. [ng/µL]", fontsize=10, color=dna_obs_color
                    )
                else:
                    twin_ax.set_ylabel("")

            # Set colors for the axis labels
            main_ax.tick_params(axis="y", colors=intact_obs_color)
            if has_dna_data:
                twin_ax.tick_params(axis="y", colors=dna_obs_color)

            # Only add x-label for bottom row
            if (has_bottom_row and i >= n_top) or (not has_bottom_row):
                main_ax.set_xlabel("Process Step", fontsize=10)
            else:
                main_ax.set_xlabel("")

        # Common legend
        intact_obs_handle = Line2D(
            [],
            [],
            marker="o",
            color=intact_obs_color,
            linestyle="-",
            markersize=6,
            label="Intact (Obs)",
        )
        intact_pred_handle = Line2D(
            [],
            [],
            marker="x",
            color=intact_pred_color,
            linestyle="--",
            markersize=6,
            label="Intact (Pred)",
        )
        dna_obs_handle = Line2D(
            [],
            [],
            marker="s",
            color=dna_obs_color,
            linestyle="-",
            markersize=6,
            label="DNA (Obs)",
        )
        dna_pred_handle = Line2D(
            [],
            [],
            marker="d",
            color=dna_pred_color,
            linestyle="--",
            markersize=6,
            label="DNA (Pred)",
        )

        fig.legend(
            handles=[
                intact_obs_handle,
                intact_pred_handle,
                dna_obs_handle,
                dna_pred_handle,
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - (top_margin / 3) / fig_height),
            ncol=4,
            fontsize=10,
        )

        # Add global title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)

        # Save figure
        scale_suffix = "log" if log_scale_dna else "linear"
        filename = f"combined_intact_dna_vs_process_{scale_suffix}.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_yield_contour(self, k: float, alpha: float, F0: float = 1.0, subdir: str = None):
        """
        Generate a contour plot of predicted yield vs. pressure and number of passes.

        Args:
            k: Fitted k parameter
            alpha: Fitted alpha parameter
            F0: Initial intact fraction
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))

        # Generate grid data
        pressures = np.linspace(400, 1200, 100)
        passes = np.linspace(0, 6, 100)
        P, N = np.meshgrid(pressures, passes)

        # Calculate predicted yield
        P_safe = np.maximum(P, 1e-9)  # Prevent numeric issues
        D = k * (P_safe ** alpha) * N
        yield_pred = 1.0 - (F0 * np.exp(-D))
        yield_pred = np.clip(yield_pred, 0, 1)

        # Create contour plot
        levels = np.linspace(0, 1, 21)
        contour = ax.contourf(P, N, yield_pred, levels=levels, cmap='viridis', extend='neither')
        contour_lines = ax.contour(P, N, yield_pred, levels=levels[::2], colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

        # Labels and formatting
        ax.set_xlabel("Pressure (bar)", fontsize=12)
        ax.set_ylabel("Number of Passes (N)", fontsize=12)

        biomass_type_label = "Fresh Biomass (F0=1.0)" if abs(F0 - 1.0) < 1e-6 else f"Frozen Biomass (F0={F0:.2f})"
        ax.set_title(f"Predicted Yield Contour Plot ({biomass_type_label})", fontsize=14, pad=15)

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, ticks=np.linspace(0, 1, 11))
        cbar.set_label(f"Predicted Yield (1 - F)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

        # Save figure
        f0_str = f"{F0:.2f}".replace('.', 'p')
        filename = f"yield_contour_plot_F0_{f0_str}.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_cv_test_predictions(self, df_test: pd.DataFrame, exp_id: int, subdir: str = None):
        """
        Plot test set predictions against actual data for a single cross-validation fold.

        Args:
            df_test: DataFrame with test predictions
            exp_id: Experiment ID for the test fold
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        # Create subdirectory for CV plots if not specified
        if not subdir:
            subdir = "cv_test_predictions"

        # Check required columns for intact fraction
        intact_required = ['experiment_id', 'process_step', 'observed_frac', 'intact_frac_pred']
        has_intact = all(col in df_test.columns for col in intact_required)

        # Check required columns for DNA predictions
        dna_required = ['experiment_id', 'process_step', 'dna_conc', 'dna_pred']
        has_dna = all(col in df_test.columns for col in dna_required)

        if not has_intact and not has_dna:
            print(f"Missing required columns for CV test prediction plot. Cannot generate plot.")
            return None

        # Create figure with 1 or 2 subplots based on available data
        n_plots = sum([has_intact, has_dna])
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True)

        # Convert to list for consistent indexing if only one subplot
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot intact fraction vs process step
        if has_intact:
            ax = axes[plot_idx]

            # Filter data to ensure it's for this experiment
            df_intact = df_test[df_test['experiment_id'] == exp_id].copy()

            if df_intact.empty:
                print(f"No intact fraction data found for experiment {exp_id}")
            else:
                # Sort by process step (assuming chronological order)
                df_intact = df_intact.sort_index()

                # Get x-axis indices and process steps
                x_indices = range(len(df_intact))
                process_steps = df_intact['process_step'].values

                # Plot observed data
                valid_obs = df_intact['observed_frac'].notna()
                if valid_obs.any():
                    # Get the indices where observed values are valid
                    obs_indices = [i for i, is_valid in enumerate(valid_obs) if is_valid]
                    ax.plot(
                        obs_indices,
                        df_intact.loc[valid_obs, 'observed_frac'],
                        marker='o',
                        linestyle='-',
                        color=self.color_schemes['observed'],
                        label='Observed'
                    )

                # Plot predicted data
                valid_pred = df_intact['intact_frac_pred'].notna()
                if valid_pred.any():
                    # Get the indices where predicted values are valid
                    pred_indices = [i for i, is_valid in enumerate(valid_pred) if is_valid]
                    ax.plot(
                        pred_indices,
                        df_intact.loc[valid_pred, 'intact_frac_pred'],
                        marker='x',
                        linestyle='--',
                        color=self.color_schemes['predicted'],
                        label='Predicted'
                    )

                # Set x-tick labels to process steps
                ax.set_xticks(x_indices)
                ax.set_xticklabels(process_steps, rotation=45, ha='right')

                # Set labels and title
                ax.set_ylabel('Fraction Intact')
                ax.set_title(f'Experiment {exp_id}: Intact Fraction Test Predictions', fontsize=12)

                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

                # Set y-axis limits with small padding
                ax.set_ylim(-0.05, 1.05)

                # Add biomass and wash procedure info as text
                if 'biomass_type' in df_intact.columns and 'wash_procedure' in df_intact.columns:
                    biomass = df_intact['biomass_type'].iloc[0]
                    wash = df_intact['wash_procedure'].iloc[0]
                    ax.text(
                        0.01, 0.98,
                        f"Biomass: {biomass}\nWash: {wash}",
                        transform=ax.transAxes,
                        va='top',
                        ha='left',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
                    )

            plot_idx += 1

        # Plot DNA concentration vs process step
        if has_dna:
            ax = axes[plot_idx]

            # Filter data to ensure it's for this experiment
            df_dna = df_test[df_test['experiment_id'] == exp_id].copy()

            if df_dna.empty:
                print(f"No DNA data found for experiment {exp_id}")
            else:
                # Sort by process step (assuming chronological order)
                df_dna = df_dna.sort_index()

                # Get x-axis indices and process steps
                x_indices = range(len(df_dna))
                process_steps = df_dna['process_step'].values

                # Filter out zero or negative values for log scale
                valid_obs = df_dna['dna_conc'] > 0
                valid_pred = df_dna['dna_pred'] > 0

                # Plot observed data
                if valid_obs.any():
                    # Get the indices where observed values are valid
                    obs_indices = [i for i, is_valid in enumerate(valid_obs) if is_valid]
                    obs_values = df_dna.loc[valid_obs, 'dna_conc'].values

                    ax.scatter(
                        obs_indices,
                        obs_values,
                        marker='o',
                        color=self.color_schemes['observed'],
                        label='Observed',
                        s=50,
                        zorder=5
                    )

                    if len(obs_indices) > 1:
                        ax.plot(
                            obs_indices,
                            obs_values,
                            linestyle='-',
                            color=self.color_schemes['observed'],
                            alpha=0.5,
                            zorder=4
                        )

                # Plot predicted data
                if valid_pred.any():
                    # Get the indices where predicted values are valid
                    pred_indices = [i for i, is_valid in enumerate(valid_pred) if is_valid]
                    pred_values = df_dna.loc[valid_pred, 'dna_pred'].values

                    ax.scatter(
                        pred_indices,
                        pred_values,
                        marker='x',
                        color=self.color_schemes['predicted'],
                        label='Predicted',
                        s=50,
                        zorder=5
                    )

                    if len(pred_indices) > 1:
                        ax.plot(
                            pred_indices,
                            pred_values,
                            linestyle='--',
                            color=self.color_schemes['predicted'],
                            alpha=0.5,
                            zorder=4
                        )

                # Set x-tick labels to process steps
                ax.set_xticks(x_indices)
                ax.set_xticklabels(process_steps, rotation=45, ha='right')

                # Set logarithmic scale for y-axis if we have valid data
                if valid_obs.any() or valid_pred.any():
                    ax.set_yscale('log')

                    # Format tick labels for log scale
                    ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
                    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

                # Set labels and title
                ax.set_ylabel('DNA Concentration [ng/µL] (Log Scale)')
                ax.set_title(f'Experiment {exp_id}: DNA Concentration Test Predictions', fontsize=12)

                # Add grid and legend
                ax.grid(True, which='major', linestyle='-', alpha=0.6)
                ax.grid(True, which='minor', linestyle=':', alpha=0.3)
                ax.legend()

        # Add common title
        plt.suptitle(f'Cross-Validation Test Predictions for Experiment {exp_id}', fontsize=14, y=1.02)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"cv_test_predictions_exp_{exp_id}.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / subdir / filename

    def create_comparison_chart(self, data: Dict[str, Dict[str, float]], title: str = "Model Comparison",
                                subdir: str = None):
        """
        Create a bar chart comparing model metrics.

        Args:
            data: Dictionary with model names as keys and metric dictionaries as values
            title: Chart title
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        if not data:
            print("No data provided for comparison chart.")
            return None

        # Extract metrics for comparison
        metrics = []
        models = []

        for model_name, metrics_dict in data.items():
            models.append(model_name)
            for metric_name, metric_value in metrics_dict.items():
                if metric_name not in metrics:
                    metrics.append(metric_name)

        # Set up figure
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=False)
        if len(metrics) == 1:
            axes = [axes]  # Make axes iterable for single metric

        # Color palette
        colors = self.color_schemes['sequential'][:len(models)]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Extract values for this metric
            values = []
            for model in models:
                values.append(data[model].get(metric, np.nan))

            # Create bar plot
            bars = ax.bar(models, values, color=colors)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    formatter = '{:.4f}' if metric.startswith('R²') else '{:.2f}'
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height * 1.01,
                        formatter.format(height),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )

            # Set labels and title
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')

            # Set ylim to start at 0 and have a small margin above max value
            if not all(pd.isna(values)):
                max_val = max([v for v in values if pd.notna(v)], default=1)
                ax.set_ylim(0, max_val * 1.15)

        # Set global title
        fig.suptitle(title, fontsize=14, y=0.98)

        # Save figure
        filename = "model_comparison_chart.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_residuals(self, df: pd.DataFrame, y_col: str, y_pred_col: str,
                       x_col: str = None, subdir: str = None,
                       title: str = None, log_scale: bool = False):
        """
        Create residual plots for model diagnostics.

        Args:
            df: DataFrame with observed and predicted values
            y_col: Column name for observed values
            y_pred_col: Column name for predicted values
            x_col: Optional column to plot residuals against
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override
            log_scale: Whether to use log scale for values

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = [y_col, y_pred_col]
        if x_col:
            required_cols.append(x_col)

        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for residual plot: {missing}")
            return None

        # Prepare data
        plot_data = df.dropna(subset=required_cols).copy()
        if plot_data.empty:
            print("Warning: No valid data for residual plot.")
            return None

        # Calculate residuals
        plot_data['residual'] = plot_data[y_col] - plot_data[y_pred_col]

        if log_scale:
            # For log scale, calculate relative residuals
            valid_mask = (plot_data[y_col] > 0) & (plot_data[y_pred_col] > 0)
            plot_data = plot_data[valid_mask].copy()

            if plot_data.empty:
                print("Warning: No valid positive data for log-scale residual plot.")
                return None

            plot_data['relative_residual'] = plot_data['residual'] / plot_data[y_col]
            residual_col = 'relative_residual'
            residual_label = 'Relative Residual'
        else:
            residual_col = 'residual'
            residual_label = 'Residual'

        # Set up figure for multiple plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Residuals vs predicted
        ax1 = axes[0]
        sns.scatterplot(
            data=plot_data,
            x=y_pred_col,
            y=residual_col,
            hue='biomass_type' if 'biomass_type' in plot_data.columns else None,
            palette=self.color_schemes['sequential'],
            alpha=0.7,
            edgecolor='k',
            ax=ax1
        )

        # Add horizontal line at 0
        ax1.axhline(y=0, color='r', linestyle='--')

        # Labels
        pred_label = y_pred_col.replace('_', ' ').title()
        ax1.set_xlabel(f"{pred_label}")
        ax1.set_ylabel(residual_label)
        ax1.set_title(f"{residual_label} vs {pred_label}")

        if log_scale and y_pred_col in ['dna_pred']:
            ax1.set_xscale('log')

        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Residual distribution
        ax2 = axes[1]
        sns.histplot(
            data=plot_data,
            x=residual_col,
            kde=True,
            bins=20,
            color=self.color_schemes['primary'],
            ax=ax2
        )

        # Add vertical line at 0
        ax2.axvline(x=0, color='r', linestyle='--')

        # Labels
        ax2.set_xlabel(residual_label)
        ax2.set_ylabel("Count")
        ax2.set_title(f"Distribution of {residual_label}s")

        ax2.grid(True, linestyle='--', alpha=0.6)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, y=1.05)
        else:
            fig.suptitle(f"{y_col.replace('_', ' ').title()} Model Diagnostic Plots", fontsize=14, y=1.05)

        # Save figure
        y_type = y_col.split('_')[0]
        filename = f"{y_type}_residual_plots{'_log' if log_scale else ''}.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_overview_observed_vs_dose(df_raw_subset: pd.DataFrame, best_k: float, best_alpha: float, output_dir: Path,
                                       config_name: str):
        """ Plots ONLY OBSERVED data points vs. cumulative dose, using manual layout. """
        required_cols = ['experiment_id', 'observed_frac', 'total_passages_650', 'total_passages_1000',
                         'wash_procedure', 'biomass_type']
        if not all(col in df_raw_subset.columns for col in required_cols):
            missing = set(required_cols) - set(df_raw_subset.columns);
            print(f"Warning: Skipping plot_overview_observed_vs_dose missing: {missing}");
            return

        # Corrected: Filter NA experiment_id before unique()
        df_plot_data = df_raw_subset.dropna(subset=['experiment_id']).copy()
        if df_plot_data.empty: print(
            f"Warning: No experiments after dropping NA experiment_id for plot_overview_observed_vs_dose ({config_name})."); return
        experiment_ids = sorted(df_plot_data['experiment_id'].unique())
        n_total = len(experiment_ids)
        if n_total == 0: print("Warning: No experiments found for plot_overview_observed_vs_dose."); return
        if n_total > 7:
            print(
                f"Warning: plot_overview_observed_vs_dose layout (3+4) fixed for 7. Found {n_total}. Plotting first 7."); experiment_ids = experiment_ids[
                                                                                                                                           :7]; n_total = 7
        elif n_total < 7:
            print(
                f"Warning: plot_overview_observed_vs_dose layout fixed for 7. Found {n_total}. Layout might look sparse.")

        subplot_width = 2;
        subplot_height = 2;
        h_spacing = 0.5;
        v_spacing = 1.5;
        left_margin = 1;
        right_margin = 1;
        bottom_margin = 1;
        top_margin = 1.5
        n_top = 3;
        n_bottom = 4;
        has_top_row = n_total >= 1;
        has_bottom_row = n_total > 3
        total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing;
        total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
        fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
        if has_top_row and has_bottom_row:
            fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
        elif has_top_row or has_bottom_row:
            fig_height = top_margin + subplot_height + bottom_margin
        else:
            fig_height = top_margin + bottom_margin
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        bottom_y = bottom_margin / fig_height;
        bottom_h = subplot_height / fig_height
        top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
        top_h = subplot_height / fig_height
        top_row_content_width = fig_width - left_margin - right_margin;
        bottom_row_content_width = fig_width - left_margin - right_margin
        top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
        bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
        axes_top = [];
        axes_bottom = []
        if has_top_row:
            for i in range(n_top): ax_left = (top_left_offset + i * (
                        subplot_width + h_spacing)) / fig_width; axes_top.append(
                fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))
        if has_bottom_row:
            for i in range(n_bottom): ax_left = (bottom_left_offset + i * (
                        subplot_width + h_spacing)) / fig_width; axes_bottom.append(
                fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))
        all_axes = axes_top + axes_bottom
        top_experiments = experiment_ids[:min(n_top, n_total)];
        bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]

        for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)):
            # Use df_plot_data which has NAs dropped
            exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
            if exp_data.empty: continue
            x_vals = compute_cumulative_dose_values(exp_data, best_k, best_alpha)
            ax.plot(x_vals, exp_data['observed_frac'], marker='o', linestyle='-')
            ax.set_xlabel('');
            if j == 0:
                ax.set_ylabel("Observed Fraction Intact", fontsize=10)
            else:
                ax.set_ylabel('')
            row0 = exp_data.iloc[0];
            wash = row0['wash_procedure'];
            biomass = row0['biomass_type']
            ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
            ax.set_title(f'Experiment {exp_id}', fontsize=10);
            ax.set_ylim(0, 1.05);
            ax.grid(True, linestyle=':', alpha=0.6)

        for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)):
            exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
            if exp_data.empty: continue
            x_vals = compute_cumulative_dose_values(exp_data, best_k, best_alpha)
            ax.plot(x_vals, exp_data['observed_frac'], marker='o', linestyle='-')
            ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel("Observed Fraction Intact", fontsize=10)
            else:
                ax.set_ylabel('')
            row0 = exp_data.iloc[0];
            wash = row0['wash_procedure'];
            biomass = row0['biomass_type']
            ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
            ax.set_title(f'Experiment {exp_id}', fontsize=10);
            ax.set_ylim(0, 1.05);
            ax.grid(True, linestyle=':', alpha=0.6)

        for k in range(n_total, 7):
            if k < len(all_axes): all_axes[k].axis('off')
        fig.text(0.5, (bottom_margin * 0.4) / fig_height, "Cumulative Dose", ha='center', va='center', fontsize=12)
        fig.suptitle("Overview: Observed Fraction Intact vs. Cumulative Dose", fontsize=14,
                     y=1.0 - (top_margin * 0.3) / fig_height)

        output_filename = f"overview_observed_vs_dose.png"
        output_path = output_dir / output_filename;
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_path, dpi=300); print(f"  Saved observed data overview plot to {output_path}")
        except Exception as e:
            print(f"  Error saving observed data overview plot: {e}")
        plt.close(fig)

    # --- Plot 5: Observed vs Step Overview (Manual Layout & sns.lineplot) ---
    # (Using the finalized version from previous steps)
    def plot_overview_observed_vs_step(df_raw_subset: pd.DataFrame, output_dir: Path, config_name: str):
        """ Plots ONLY OBSERVED data points vs. process_step, using manual layout and sns.lineplot. """
        required_cols = ['experiment_id', 'process_step', 'observed_frac', 'wash_procedure', 'biomass_type']
        if not all(col in df_raw_subset.columns for col in required_cols):
            missing = set(required_cols) - set(df_raw_subset.columns);
            print(f"Warning: Skipping plot_overview_observed_vs_step missing: {missing}");
            return

        # Corrected: Filter NA experiment_id before unique()
        df_plot_data = df_raw_subset.dropna(subset=['experiment_id']).copy()
        if df_plot_data.empty: print(
            f"Warning: No experiments after dropping NA experiment_id for plot_overview_observed_vs_step ({config_name})."); return
        experiment_ids = sorted(df_plot_data['experiment_id'].unique())
        n_total = len(experiment_ids)
        if n_total == 0: print("Warning: No experiments found for plot_overview_observed_vs_step."); return
        if n_total > 7:
            print(
                f"Warning: plot_overview_observed_vs_step layout (3+4) fixed for 7. Found {n_total}. Plotting first 7."); experiment_ids = experiment_ids[
                                                                                                                                           :7]; n_total = 7
        elif n_total < 7:
            print(
                f"Warning: plot_overview_observed_vs_step layout fixed for 7. Found {n_total}. Layout might look sparse.")

        subplot_width = 2;
        subplot_height = 2;
        h_spacing = 0.5;
        v_spacing = 1.5;
        left_margin = 1;
        right_margin = 1;
        bottom_margin = 1;
        top_margin = 1.5
        n_top = 3;
        n_bottom = 4;
        has_top_row = n_total >= 1;
        has_bottom_row = n_total > 3
        total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing;
        total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
        fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
        if has_top_row and has_bottom_row:
            fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
        elif has_top_row or has_bottom_row:
            fig_height = top_margin + subplot_height + bottom_margin
        else:
            fig_height = top_margin + bottom_margin
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        bottom_y = bottom_margin / fig_height;
        bottom_h = subplot_height / fig_height
        top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
        top_h = subplot_height / fig_height
        top_row_content_width = fig_width - left_margin - right_margin;
        bottom_row_content_width = fig_width - left_margin - right_margin
        top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
        bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
        axes_top = [];
        axes_bottom = []
        if has_top_row:
            for i in range(n_top): ax_left = (top_left_offset + i * (
                        subplot_width + h_spacing)) / fig_width; axes_top.append(
                fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))
        if has_bottom_row:
            for i in range(n_bottom): ax_left = (bottom_left_offset + i * (
                        subplot_width + h_spacing)) / fig_width; axes_bottom.append(
                fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))
        all_axes = axes_top + axes_bottom
        top_experiments = experiment_ids[:min(n_top, n_total)];
        bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]

        for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)):
            exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
            if exp_data.empty: continue
            sns.lineplot(data=exp_data, x='process_step', y='observed_frac', marker='o', ax=ax, sort=False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9);
            ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel("Observed Fraction Intact", fontsize=10)
            else:
                ax.set_ylabel('')
            row0 = exp_data.iloc[0];
            wash = row0['wash_procedure'];
            biomass = row0['biomass_type']
            ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
            ax.set_title(f'Experiment {exp_id}', fontsize=10);
            ax.set_ylim(0, 1.05);
            ax.grid(True, linestyle=':', alpha=0.6)

        for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)):
            exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
            if exp_data.empty: continue
            sns.lineplot(data=exp_data, x='process_step', y='observed_frac', marker='o', ax=ax, sort=False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9);
            ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel("Observed Fraction Intact", fontsize=10)
            else:
                ax.set_ylabel('')
            row0 = exp_data.iloc[0];
            wash = row0['wash_procedure'];
            biomass = row0['biomass_type']
            ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
            ax.set_title(f'Experiment {exp_id}', fontsize=10);
            ax.set_ylim(0, 1.05);
            ax.grid(True, linestyle=':', alpha=0.6)

        for k in range(n_total, 7):
            if k < len(all_axes): all_axes[k].axis('off')
        fig.text(0.5, (bottom_margin * 0.4) / fig_height, "Process Step", ha='center', va='center', fontsize=12)
        fig.suptitle("Overview: Observed Fraction Intact over Process Steps", fontsize=14,
                     y=1.0 - (top_margin * 0.3) / fig_height)

        output_filename = f"overview_observed_vs_process_step.png"
        output_path = output_dir / output_filename;
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(
                f"  Saved observed data vs process step overview plot to {output_path}")
        except Exception as e:
            print(f"  Error saving observed data vs process step overview plot: {e}")
        plt.close(fig)

    def plot_dna_vs_step(self, df: pd.DataFrame, config_name: str = "default", subdir: str = None):
        """
        Plot DNA concentration vs process steps for each experiment.

        Args:
            df: DataFrame with DNA concentrations, process steps, and experiment IDs
            config_name: Name for configuration (used in title/filename)
            subdir: Optional subdirectory within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['experiment_id', 'process_step', 'dna_conc', 'dna_pred']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for DNA vs step plot: {missing}")
            return None

        # Get unique experiment IDs
        exp_ids = sorted([x for x in df['experiment_id'].unique() if pd.notna(x)])
        n_exps = len(exp_ids)

        if n_exps == 0:
            print("No experiments found for DNA vs step plot.")
            return None

        # Define step order for sorting
        step_order = {
            'resuspended biomass': 0,
            'initial lysis': 1,
            '1st wash': 2,
            '2nd wash': 3,
            '3rd wash': 4,
            '4th wash': 5
        }

        # Create figure with subplots (2 rows if more than 3 experiments)
        rows = 2 if n_exps > 3 else 1
        cols = (n_exps + rows - 1) // rows

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        # Plot each experiment
        for i, exp_id in enumerate(exp_ids):
            if i >= len(axes):
                break

            ax = axes[i]
            exp_data = df[df['experiment_id'] == exp_id].copy()

            # Sort by process step
            exp_data['step_order'] = exp_data['process_step'].map(
                lambda x: step_order.get(str(x).lower(), 999))
            exp_data = exp_data.sort_values('step_order')

            # Get process steps for x-axis
            x_steps = exp_data['process_step'].values

            # Plot observed values
            valid_obs = exp_data['dna_conc'] > 0
            if valid_obs.any():
                ax.plot(range(len(x_steps)), exp_data.loc[valid_obs, 'dna_conc'],
                        'o-', color='blue', markersize=8, label='Observed')

            # Plot predicted values
            valid_pred = exp_data['dna_pred'] > 0
            if valid_pred.any():
                ax.plot(range(len(x_steps)), exp_data.loc[valid_pred, 'dna_pred'],
                        'x--', color='red', markersize=8, label='Predicted')

            # Set log scale for y-axis
            ax.set_yscale('log')

            # Set x-axis labels
            ax.set_xticks(range(len(x_steps)))
            ax.set_xticklabels(x_steps, rotation=45, ha='right')

            # Get biomass type and wash procedure for title
            if 'biomass_type' in exp_data.columns and 'wash_procedure' in exp_data.columns:
                biomass = exp_data['biomass_type'].iloc[0]
                wash = exp_data['wash_procedure'].iloc[0]
                ax.set_title(f'Exp {exp_id}: {biomass}, {wash}')
            else:
                ax.set_title(f'Experiment {exp_id}')

            # Add labels
            if i % cols == 0:  # First column
                ax.set_ylabel('DNA Concentration [ng/µL]')

            # Add legend
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_exps, len(axes)):
            axes[i].axis('off')

        # Add title
        plt.suptitle(f'DNA Concentration vs Process Steps ({config_name})', fontsize=16, y=1.02)
        plt.tight_layout()

        # Save figure
        filename = f"dna_vs_process_steps_{config_name}.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename
