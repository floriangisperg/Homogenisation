# src/analysis/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Dict, Any, Tuple, Optional, List, Union


class VisualizationManager:
    """
    Manages the generation of all visualizations for the lysis analysis.
    Provides a consistent interface and styling for plots.
    """

    def __init__(self, output_dir: Path, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize the visualization manager.

        Args:
            output_dir: Base directory for saving plots
            style: Matplotlib style to use
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('ggplot')  # Fallback style

        # Set default font to one with better Unicode support
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Verdana', 'Arial']
        except Exception as e:
            print(f"Warning: Could not set default font: {e}")

        # Color schemes
        self.color_schemes = {
            'primary': 'royalblue',
            'secondary': 'crimson',
            'tertiary': 'forestgreen',
            'observed': 'blue',
            'predicted': 'red',
            'sequential': sns.color_palette("Set2")
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

    def plot_intact_parity(self, df: pd.DataFrame, subdir: str = None, title: str = None):
        """
        Create parity plot for intact fraction predictions.

        Args:
            df: DataFrame with 'observed_frac', 'intact_frac_pred', etc.
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['observed_frac', 'intact_frac_pred', 'wash_procedure', 'biomass_type']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for intact parity plot: {missing}")
            return None

        plot_data = df.dropna(subset=required_cols).copy()
        if plot_data.empty:
            print("Warning: No valid data for intact parity plot.")
            return None

        # Set up figure
        fig, ax = plt.subplots(figsize=(7, 6))

        # Define markers and colors
        markers = {"fresh biomass": "o", "frozen biomass": "s"}
        wash_types = plot_data["wash_procedure"].unique()
        palette = self.color_schemes['sequential'][:len(wash_types)]
        color_map = dict(zip(wash_types, palette))

        # Plot data points
        legend_elements = []
        plotted_labels = set()

        for wash in wash_types:
            wash_color = color_map.get(wash, 'black')
            for biomass_key, biomass_marker in markers.items():
                subset = plot_data[(plot_data["wash_procedure"] == wash) &
                                   (plot_data["biomass_type"] == biomass_key)]
                if not subset.empty:
                    ax.scatter(subset["observed_frac"], subset["intact_frac_pred"],
                               marker=biomass_marker, color=wash_color,
                               edgecolor="k", s=50, alpha=0.9)

                    # Add to legend if not already there
                    label = f"{wash}, {biomass_key}"
                    if label not in plotted_labels:
                        legend_elements.append(
                            Line2D([0], [0], marker=biomass_marker, color='w',
                                   label=label, markerfacecolor=wash_color,
                                   markeredgecolor='k', markersize=7)
                        )
                        plotted_labels.add(label)

        # Plot y=x line
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', lw=1.5)
        legend_elements.append(Line2D([0], [0], color='k', linestyle='--', lw=1.5, label='y = x'))

        # Labels and formatting
        ax.set_xlabel("Observed Fraction Intact", fontsize=12)
        ax.set_ylabel("Predicted Fraction Intact", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, pad=15)
        else:
            ax.set_title("Parity Plot: Observed vs. Predicted Intact Fraction", fontsize=14, pad=15)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', labelsize=10)

        # Legend
        ax.legend(handles=legend_elements, fontsize=9, title="Condition",
                  title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

        # Save figure
        filename = "intact_parity_plot.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_dna_parity(self, df: pd.DataFrame, subdir: str = None, title: str = None):
        """
        Create parity plot for DNA concentration predictions (log-log scale).

        Args:
            df: DataFrame with 'dna_conc', 'dna_pred', etc.
            subdir: Optional subdirectory to save within output_dir
            title: Optional title override

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['dna_conc', 'dna_pred', 'wash_procedure', 'biomass_type', 'process_step']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for DNA parity plot: {missing}")
            return None

        # Filter data and remove negative or zero values (incompatible with log scale)
        plot_data_base = df.dropna(subset=['dna_conc', 'dna_pred']).copy()
        plot_data_base = plot_data_base[plot_data_base['process_step'].str.lower() != "resuspended biomass"]
        plot_data = plot_data_base[(plot_data_base['dna_conc'] > 0) & (plot_data_base['dna_pred'] > 0)].copy()

        if plot_data.empty:
            print(f"Warning: No valid positive data points found for DNA parity plot. Skipping.")
            return None

        # Set up figure
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        # Add user-friendly column names for the legend
        plot_data['Wash Type'] = plot_data['wash_procedure'].str.replace(" wash", "")
        plot_data['Biomass'] = plot_data['biomass_type'].str.replace(" biomass", "").str.capitalize()

        # Create scatterplot
        sns.scatterplot(
            data=plot_data, x="dna_conc", y="dna_pred",
            hue="Wash Type", style="Biomass",
            markers={"Fresh": "o", "Frozen": "s"},
            palette=self.color_schemes['sequential'],
            s=50, edgecolor="k", alpha=0.8, ax=ax
        )

        # Set log scales
        ax.set(xscale="log", yscale="log")

        # Plot y=x line
        min_val = min(plot_data["dna_conc"].min(), plot_data["dna_pred"].min()) * 0.8
        max_val = max(plot_data["dna_conc"].max(), plot_data["dna_pred"].max()) * 1.2
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, zorder=1)

        # Labels and formatting
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("Observed DNA [ng/µL] (Log Scale)", fontsize=11)
        ax.set_ylabel("Predicted DNA [ng/µL] (Log Scale)", fontsize=11)

        if title:
            ax.set_title(title, fontsize=13, pad=15)
        else:
            ax.set_title("DNA Concentration Parity Plot (Log-Log Scale)", fontsize=13, pad=15)

        # Format tick labels for log scale
        ax.xaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
        ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis='both', labelsize=9)

        # Legend and grid
        plt.legend(title="Condition", title_fontsize=10, fontsize=9,
                   bbox_to_anchor=(1.03, 1), loc='upper left')
        ax.grid(True, which='major', linestyle='-', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)

        # Save figure
        filename = "dna_parity_plot_loglog.png"
        self._save_plot(fig, filename, subdir)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

    def plot_overview_fitted(self, df: pd.DataFrame, k: float, alpha: float, subdir: str = None):
        """
        Generate overview plot showing both observed and fitted intact fractions vs dose.

        Args:
            df: DataFrame with intact fraction data
            k: Fitted k parameter
            alpha: Fitted alpha parameter
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['experiment_id', 'observed_frac', 'cumulative_dose', 'intact_biomass_percent',
                         'biomass_type', 'wash_procedure']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for overview fitted plot: {missing}")
            return None

        df_plot_data = df.dropna(subset=['experiment_id']).copy()
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

    def plot_dna_vs_step(self, df: pd.DataFrame, config_name: str = "", subdir: str = None):
        """
        Plot DNA concentration vs process step overview using log scale Y.

        Args:
            df: DataFrame with DNA data
            config_name: Name for the configuration (used in title)
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        required_cols = ['experiment_id', 'process_step', 'dna_conc', 'dna_pred',
                         'wash_procedure', 'biomass_type']

        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            print(f"Missing columns for DNA vs step plot: {missing}")
            return None

        # Prepare data
        df_plot_data = df.dropna(subset=['experiment_id']).copy()
        if df_plot_data.empty:
            print(f"Warning: No experiments after dropping NA experiment_id for plot_dna_vs_step ({config_name}).")
            return None

        experiment_ids = sorted(df_plot_data['experiment_id'].unique())
        n_total = len(experiment_ids)
        if n_total == 0:
            print("Warning: No experiments found for plot_dna_vs_step.")
            return None

        if n_total > 7:
            print(f"Warning: plot_dna_vs_step layout (3+4) fixed for 7. Found {n_total}. Plotting first 7.")
            experiment_ids = experiment_ids[:7]
            n_total = 7
        elif n_total < 7:
            print(f"Warning: plot_dna_vs_step layout fixed for 7. Found {n_total}. Layout might look sparse.")

        # Prepare layout parameters
        subplot_width = 2
        subplot_height = 2
        h_spacing = 0.5
        v_spacing = 1.5
        left_margin = 1
        right_margin = 1
        bottom_margin = 1
        top_margin = 1.5

        n_top = 3
        n_bottom = 4
        has_top_row = n_total >= 1
        has_bottom_row = n_total > 3

        total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing
        total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing

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
        bottom_y = bottom_margin / fig_height
        bottom_h = subplot_height / fig_height
        top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
        top_h = subplot_height / fig_height

        top_row_content_width = fig_width - left_margin - right_margin
        bottom_row_content_width = fig_width - left_margin - right_margin

        top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
        bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2

        axes_top = []
        axes_bottom = []

        if has_top_row:
            for i in range(n_top):
                ax_left = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width
                axes_top.append(fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))

        if has_bottom_row:
            for i in range(n_bottom):
                ax_left = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width
                axes_bottom.append(fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))

        all_axes = axes_top + axes_bottom
        top_experiments = experiment_ids[:min(n_top, n_total)]
        bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]

        # Set colors and prepare for Y-axis limits
        obs_color = self.color_schemes['observed']
        pred_color = self.color_schemes['predicted']
        min_yaxis_val = np.inf
        max_yaxis_val = -np.inf

        # Function to plot a single experiment
        def plot_single_dna_step(ax, exp_id_single, df_plot):
            nonlocal min_yaxis_val, max_yaxis_val

            exp_data = df_plot[df_plot['experiment_id'] == exp_id_single].sort_index()
            if exp_data.empty:
                return

            # Get valid data points and update Y-axis limits
            obs_plot = exp_data[exp_data['dna_conc'] > 0]
            pred_plot = exp_data[exp_data['dna_pred'] > 0]

            if not obs_plot.empty:
                min_yaxis_val = min(min_yaxis_val, obs_plot['dna_conc'].min())
                max_yaxis_val = max(max_yaxis_val, obs_plot['dna_conc'].max())

            if not pred_plot.empty:
                min_yaxis_val = min(min_yaxis_val, pred_plot['dna_pred'].min())
                max_yaxis_val = max(max_yaxis_val, pred_plot['dna_pred'].max())

            # Plot observed and predicted data
            sns.lineplot(
                data=exp_data,
                x='process_step',
                y='dna_conc',
                marker='o',
                ax=ax,
                label='Observed',
                color=obs_color,
                sort=False,
                legend=False
            )

            sns.lineplot(
                data=exp_data,
                x='process_step',
                y='dna_pred',
                marker='x',
                linestyle='--',
                ax=ax,
                label='Predicted',
                color=pred_color,
                sort=False,
                legend=False
            )

            # Set log scale and format axes
            ax.set_yscale('log')
            ax.set_xlabel('')
            ax.set_ylabel('')

            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
            plt.setp(ax.get_yticklabels(), fontsize=9)

            # Add experiment info
            row0 = exp_data.iloc[0]
            wash = row0['wash_procedure']
            biomass = row0['biomass_type']

            ax.text(
                0.95, 0.95,
                f"{wash}\n{biomass}",
                transform=ax.transAxes,
                va='top',
                ha='right',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
            )

            ax.set_title(f'Experiment {exp_id_single}', fontsize=10)
            ax.grid(True, which='major', linestyle='-', alpha=0.6)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)

        # Plot each experiment
        for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)):
            plot_single_dna_step(ax, exp_id, df_plot_data)
            plt.setp(ax.get_yticklabels(), visible=(j == 0))

        for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)):
            plot_single_dna_step(ax, exp_id, df_plot_data)
            plt.setp(ax.get_yticklabels(), visible=(j == 0))
            plt.setp(ax.get_xticklabels(), visible=True)  # Show x-labels for bottom row

        # Set common Y-axis range
        if np.isfinite(min_yaxis_val) and np.isfinite(max_yaxis_val) and max_yaxis_val > min_yaxis_val:
            y_min_log = np.floor(np.log10(min_yaxis_val))
            y_max_log = np.ceil(np.log10(max_yaxis_val))

            for ax in all_axes:
                if ax.has_data():
                    ax.set_ylim(10 ** y_min_log, 10 ** y_max_log)
                    ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
                    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Set Y-axis labels for leftmost plots
        if axes_top and axes_top[0].has_data():
            axes_top[0].set_ylabel("DNA [ng/µL] (Log)", fontsize=10)

        if axes_bottom and axes_bottom[0].has_data():
            axes_bottom[0].set_ylabel("DNA [ng/µL] (Log)", fontsize=10)

        # Hide empty subplots
        for k in range(n_total, 7):
            if k < len(all_axes):
                all_axes[k].axis('off')

        # Add common X-axis label and title
        fig.text(0.5, (bottom_margin * 0.4) / fig_height, "Process Step", ha='center', va='center', fontsize=12)
        fig.suptitle("Overview: DNA Concentration vs. Process Step", fontsize=14,
                     y=1.0 - (top_margin * 0.3) / fig_height)

        # Add legend
        obs_handle = Line2D([], [], marker='o', color=obs_color, linestyle='-', label='Observed')
        pred_handle = Line2D([], [], marker='x', color=pred_color, linestyle='--', label='Predicted')

        fig.legend(
            handles=[obs_handle, pred_handle],
            loc='upper center',
            bbox_to_anchor=(0.5, 1 - (top_margin / fig_height) * 0.15),
            ncol=2,
            fontsize=10
        )

        # Save figure
        filename = "overview_dna_vs_process_step.png"
        self._save_plot(fig, filename, subdir, tight_layout=False)
        plt.close(fig)

        return self.output_dir / (subdir or "") / filename

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

    def create_parameter_boxplot(self, param_data: Dict[str, List[float]], subdir: str = None):
        """
        Create boxplots for parameter distributions across LOOCV folds.

        Args:
            param_data: Dictionary with parameter names as keys and lists of values as values
            subdir: Optional subdirectory to save within output_dir

        Returns:
            Path to saved plot or None if failed
        """
        if not param_data:
            print("No parameter data provided for boxplot.")
            return None

        # Filter out parameters with insufficient data
        valid_params = {k: v for k, v in param_data.items() if len([x for x in v if pd.notna(x)]) > 1}

        if not valid_params:
            print("No valid parameter data for boxplot.")
            return None

        # Set up figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplot
        box = ax.boxplot(
            list(valid_params.values()),
            labels=list(valid_params.keys()),
            patch_artist=True,
            notch=True,
            showfliers=True
        )

        # Customize boxplot colors
        for patch in box['boxes']:
            patch.set_facecolor(self.color_schemes['primary'])
            patch.set_alpha(0.7)

        for whisker in box['whiskers']:
            whisker.set(color='black', linewidth=1.2, linestyle='-')

        for cap in box['caps']:
            cap.set(color='black', linewidth=1.2)

        for median in box['medians']:
            median.set(color='darkred', linewidth=2)

        for flier in box['fliers']:
            flier.set(marker='o', markersize=6, alpha=0.7)

        # Labels and grid
        ax.set_title("Parameter Variation Across LOOCV Folds", fontsize=14, pad=20)
        ax.set_ylabel("Parameter Value", fontsize=12)
        ax.set_xlabel("Parameter", fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Rotate x-tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        # Add mean values as text
        for i, (param, values) in enumerate(valid_params.items()):
            valid_values = [v for v in values if pd.notna(v)]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            # Format based on parameter type
            if param == 'k':
                fmt = "{:.3e} ± {:.3e}"
            elif param in ['C_release_fresh', 'C_release_frozen']:
                fmt = "{:.1f} ± {:.1f}"
            else:
                fmt = "{:.3f} ± {:.3f}"

            text = fmt.format(mean_val, std_val)
            ax.text(
                i + 1, ax.get_ylim()[0] * 1.1,
                text,
                ha='center',
                va='top',
                fontsize=9,
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

        # Save figure
        filename = "parameter_distribution_boxplot.png"
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

    def plot_parameter_sensitivity(self, sensitivity_data: List[Dict], target_params: List[str] = None,
                                   target_metrics: List[str] = None, subdir: str = None):
        """
        Create parameter sensitivity plots.

        Args:
            sensitivity_data: List of dictionaries with sensitivity analysis results
            target_params: List of parameters to plot (if None, plot all)
            target_metrics: List of metrics to show (if None, plot R² and RMSE)
            subdir: Optional subdirectory to save within output_dir

        Returns:
            List of paths to saved plots or empty list if failed
        """
        if not sensitivity_data:
            print("No sensitivity data provided.")
            return []

        # Default metrics if not specified
        if target_metrics is None:
            target_metrics = ['R²', 'RMSE']

        # Extract all parameters if target_params not specified
        if target_params is None:
            target_params = set()
            for result in sensitivity_data:
                if 'param_name' in result:
                    target_params.add(result['param_name'])
            target_params = sorted(list(target_params))

        saved_plots = []

        # Create a plot for each parameter
        for param in target_params:
            # Filter data for this parameter
            param_data = [r for r in sensitivity_data if r.get('param_name') == param]
            if not param_data:
                continue

            # Sort by parameter value
            param_data.sort(key=lambda x: x.get('param_value', 0))

            # Extract data for plotting
            x_values = [r.get('param_value') for r in param_data]

            # Verify that we have numeric values
            if not x_values or not all(isinstance(x, (int, float)) for x in x_values if pd.notna(x)):
                print(f"Skipping parameter {param}: non-numeric or empty values")
                continue

            # Get model types
            model_types = set()
            for r in param_data:
                if 'model_type' in r:
                    model_types.add(r['model_type'])

            # Set up figure
            fig, axes = plt.subplots(len(target_metrics), 1, figsize=(8, 4 * len(target_metrics)), sharex=True)
            if len(target_metrics) == 1:
                axes = [axes]  # Make axes iterable for single metric

            # Plot each metric
            for i, metric in enumerate(target_metrics):
                ax = axes[i]

                for model_type in sorted(model_types):
                    # Filter for this model type
                    model_data = [r for r in param_data if r.get('model_type') == model_type]

                    if not model_data:
                        continue

                    # Get metric values
                    y_values = [r.get('metrics', {}).get(metric) for r in model_data]

                    # Plot line
                    model_label = model_type.replace('_', ' ').title() if model_type else 'Unknown'
                    ax.plot(x_values, y_values, marker='o', linestyle='-', label=model_label)

                    # Mark the base value if available
                    for r in model_data:
                        if r.get('base_value') == r.get('param_value'):
                            ax.plot(r.get('param_value'), r.get('metrics', {}).get(metric),
                                    marker='*', markersize=12, color='red')

                # Labels and grid
                ax.set_ylabel(metric)
                ax.set_title(f"Effect of {param} on {metric}")
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

            # Set x-axis label on bottom plot
            axes[-1].set_xlabel(f"{param} Value")

            # Overall title
            fig.suptitle(f"Sensitivity Analysis: {param}", fontsize=14, y=0.98)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            filename = f"sensitivity_{param.replace('.', '_')}.png"
            save_path = self.output_dir / (subdir or "") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                plt.savefig(save_path, dpi=300)
                print(f"  Saved sensitivity plot to {save_path}")
                saved_plots.append(save_path)
            except Exception as e:
                print(f"  Error saving sensitivity plot: {e}")

            plt.close(fig)

        return saved_plots

