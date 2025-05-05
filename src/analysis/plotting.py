# src/analysis/plotting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For log ticker formatting
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict

# --- Imports from other analysis modules ---
try:
    from .data_processing import compute_cumulative_dose_values, add_cumulative_dose
    from .models.mechanistic_model import predict_intact_fraction
except ImportError:
    print("Warning: Running plotting.py potentially outside of the main script context.")
    def compute_cumulative_dose_values(df, k, a): return [0]*len(df)
    def add_cumulative_dose(df, k, a): df['cumulative_dose']=0; return df
    def predict_intact_fraction(df, k, a): df['intact_frac_pred']=df['observed_frac']; return df

# Set default font to one with better Unicode support (Optional but recommended)
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Verdana', 'Arial']
except Exception as e:
    print(f"Warning: Could not set default font to DejaVu Sans: {e}")


# --- Plotting Configuration ---
try:
    plt.style.use('seaborn-v0_8-whitegrid') # Use a grid style
except OSError:
    plt.style.use('ggplot') # Fallback

# --- Plot 1: Overview Plot (Fitted Data vs Dose - Manual Layout) ---
# (Using the finalized version from previous steps that uses fig.add_axes)
def plot_overview_fitted(df_fit: pd.DataFrame, best_k: float, best_alpha: float, output_dir: Path):
    """
    Generates a multi-panel plot showing observed points (scatter) and the
    smooth theoretical fitted model curve (line) against cumulative dose,
    using the user's original manual subplot layout.
    """
    required_cols = ['experiment_id', 'observed_frac', 'cumulative_dose', 'intact_biomass_percent',
                     'biomass_type', 'wash_procedure'] # Added intact_biomass_percent
    if not all(col in df_fit.columns for col in required_cols):
        missing = set(required_cols) - set(df_fit.columns)
        raise ValueError(f"Input DataFrame for plot_overview_fitted is missing columns: {missing}")

    df_plot_data = df_fit.dropna(subset=['experiment_id']).copy()
    if df_plot_data.empty: print("Warning: No experiments after dropping NA experiment_id for plot_overview_fitted."); return
    experiment_ids = sorted(df_plot_data['experiment_id'].unique())
    n_total = len(experiment_ids)
    if n_total == 0: print("Warning: No experiments found for plot_overview_fitted."); return

    n_top = 3 if n_total >= 3 else n_total
    n_bottom = min(4, n_total - n_top)
    has_top_row = n_top > 0; has_bottom_row = n_bottom > 0
    subplot_width = 2; subplot_height = 2; h_spacing = 0.5
    v_spacing = 1.0 if (has_top_row and has_bottom_row) else 0
    left_margin = 1; right_margin = 1; bottom_margin = 1; top_margin = 1.2
    total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing if has_top_row else 0
    total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing if has_bottom_row else 0
    fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
    if has_top_row and has_bottom_row: fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
    elif has_top_row or has_bottom_row: fig_height = top_margin + subplot_height + bottom_margin
    else: fig_height = top_margin + bottom_margin
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    axes_list = []
    if has_top_row:
        top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_margin / fig_height
        top_h = subplot_height / fig_height
        top_row_content_width = fig_width - left_margin - right_margin
        top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
        for i in range(n_top):
            ax_left_norm = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width
            ax_width_norm = subplot_width / fig_width
            axes_list.append(fig.add_axes([ax_left_norm, top_y, ax_width_norm, top_h]))
    if has_bottom_row:
        bottom_y = bottom_margin / fig_height; bottom_h = subplot_height / fig_height
        bottom_row_content_width = fig_width - left_margin - right_margin
        bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
        for i in range(n_bottom):
            ax_left_norm = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width
            ax_width_norm = subplot_width / fig_width
            axes_list.append(fig.add_axes([ax_left_norm, bottom_y, ax_width_norm, bottom_h]))

    observed_color = "royalblue"; predicted_color = "crimson"
    common_yticks = np.arange(0, 1.1, 0.25)
    max_overall_dose = df_plot_data['cumulative_dose'].max()
    xlim_max = max(2, np.ceil(max_overall_dose)) if pd.notna(max_overall_dose) else 2
    common_xlim = (-0.05 * xlim_max, xlim_max * 1.05)

    for i, (ax, exp_id) in enumerate(zip(axes_list, experiment_ids)):
        exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
        if exp_data.empty: continue

        ax.scatter(exp_data['cumulative_dose'], exp_data['observed_frac'],
                   color=observed_color, s=25, zorder=5, edgecolor='k', linewidth=0.5)

        row0 = exp_data.iloc[0]
        if row0['biomass_type'] == "fresh biomass": F0 = 1.0
        else:
            F0 = row0['intact_biomass_percent'] / 100.0
            if pd.isna(F0) or not (0 < F0 <= 1): F0 = row0['observed_frac']
            if pd.isna(F0) or not (0 < F0 <= 1): F0 = None

        if F0 is not None:
            max_dose_exp = exp_data['cumulative_dose'].max()
            if pd.isna(max_dose_exp) or max_dose_exp <= 0: max_dose_exp = 1e-6
            dose_fine = np.linspace(0, max_dose_exp, 200)
            predicted_fine = F0 * np.exp(-dose_fine)
            ax.plot(dose_fine, predicted_fine, color=predicted_color, lw=1.5, zorder=3)

        ax.set_ylim(-0.05, 1.05); ax.set_yticks(common_yticks); ax.set_xlim(common_xlim)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        is_leftmost_top = has_top_row and (i == 0)
        is_leftmost_bottom = has_bottom_row and (i == n_top)
        if is_leftmost_top or is_leftmost_bottom: ax.set_ylabel("Fraction Intact", fontsize=11)
        else: ax.set_ylabel(''); plt.setp(ax.get_yticklabels(), visible=False)

        is_on_bottom_row = has_bottom_row and (i >= n_top)
        is_only_row = not has_bottom_row
        if is_on_bottom_row or is_only_row: ax.set_xlabel('')
        else: ax.set_xlabel(''); plt.setp(ax.get_xticklabels(), visible=False)

        wash = row0['wash_procedure'] # Use full annotation
        biomass = row0['biomass_type'] # Use full annotation
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
        ax.set_title(f'Exp {exp_id}', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)

    if has_top_row or has_bottom_row:
         fig.text(0.5, (bottom_margin/2) / fig_height, r"Cumulative Dose ($k \cdot P^{\alpha} \cdot N$)",
                  ha='center', va='center', fontsize=12)

    observed_handle = Line2D([], [], marker='o', color=observed_color, linestyle='None', markersize=6, markeredgecolor='k', markeredgewidth=0.5, label='Observed')
    predicted_handle = Line2D([], [], color=predicted_color, lw=1.5, label='Model Fit')
    fig.legend(handles=[observed_handle, predicted_handle], loc='upper center',
               bbox_to_anchor=(0.5, 1 - (top_margin/fig_height)*0.2), ncol=2, fontsize=11)

    output_path = output_dir / "overview_fitted_vs_dose.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=300)
        print(f"  Saved overview plot to {output_path}")
    except Exception as e: print(f"  Error saving overview plot: {e}")
    plt.close(fig)


# --- Plot 2: Intact Parity Plot (Using User's Preferred Matplotlib Style) ---
# (Using the finalized version from previous steps)
def plot_parity(df_fit: pd.DataFrame, output_dir: Path):
    """ Parity plot for intact fraction using direct Matplotlib calls. """
    required_cols = ['observed_frac', 'intact_frac_pred', 'wash_procedure', 'biomass_type']
    if not all(col in df_fit.columns for col in required_cols):
        missing = set(required_cols) - set(df_fit.columns); raise ValueError(f"Missing columns for plot_parity: {missing}")
    plot_data = df_fit.dropna(subset=required_cols).copy()
    if plot_data.empty: print("Warning: No valid data for intact parity plot."); return

    markers = {"fresh biomass": "o", "frozen biomass": "s"}
    sns.set_style("darkgrid")
    wash_types = plot_data["wash_procedure"].unique() # Use original order
    palette = sns.color_palette("Set2", n_colors=len(wash_types))
    color_map = dict(zip(wash_types, palette))
    plt.figure(figsize=(7, 6), dpi=300)
    legend_elements = []; plotted_labels = set()

    for wash in wash_types:
        wash_color = color_map.get(wash, 'black')
        for biomass_key, biomass_marker in markers.items():
            subset = plot_data[(plot_data["wash_procedure"] == wash) & (plot_data["biomass_type"] == biomass_key)]
            if not subset.empty:
                label = f"{wash}, {biomass_key}"
                plt.scatter(subset["observed_frac"], subset["intact_frac_pred"], marker=biomass_marker,
                            color=wash_color, edgecolor="k", s=50, alpha=0.9)
                if label not in plotted_labels:
                     legend_elements.append(Line2D([0], [0], marker=biomass_marker, color='w', label=label,
                                       markerfacecolor=wash_color, markeredgecolor='k', markersize=7))
                     plotted_labels.add(label)

    lims = [0, 1]
    plt.plot(lims, lims, 'k--', lw=1.5, label="y = x"); legend_elements.append(Line2D([0], [0], color='k', linestyle='--', lw=1.5, label='y = x'))
    plt.xlabel("Observed Fraction Intact", fontsize=12); plt.ylabel("Predicted Fraction Intact", fontsize=12)
    plt.title("Parity Plot: Observed vs. Predicted", fontsize=14, pad=15); plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=10); plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)
    plt.legend(handles=legend_elements, fontsize=9, title="Condition", title_fontsize=10, bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6); plt.gca().set_aspect('equal', adjustable='box')

    output_path = output_dir / "parity_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try: plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(f"  Saved parity plot to {output_path}")
    except Exception as e: print(f"  Error saving parity plot: {e}")
    plt.close()
    # plt.style.use('default') # Reset style if needed


# --- Plot 3: Yield Contour Plot ---
# (Using the finalized version from previous steps)
def plot_yield_contour(best_k: float, best_alpha: float, output_dir: Path, F0: float = 1.0):
    """ Generates a contour plot of predicted yield vs. pressure and number of passes. """
    pressures = np.linspace(400, 1200, 100); passes = np.linspace(0, 6, 100)
    P, N = np.meshgrid(pressures, passes)
    P_safe = np.maximum(P, 1e-9); D = best_k * (P_safe ** best_alpha) * N
    yield_pred = 1.0 - (F0 * np.exp(-D)); yield_pred = np.clip(yield_pred, 0, 1)
    plt.figure(figsize=(7, 5.5), dpi=300); levels = np.linspace(0, 1, 21)
    contour = plt.contourf(P, N, yield_pred, levels=levels, cmap='viridis', extend='neither')
    contour_lines = plt.contour(P, N, yield_pred, levels=levels[::2], colors='white', linewidths=0.5, alpha=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    plt.xlabel("Pressure (bar)", fontsize=12); plt.ylabel("Number of Passes (N)", fontsize=12)
    biomass_type_label = "Fresh Biomass (F0=1.0)" if abs(F0 - 1.0) < 1e-6 else f"Frozen Biomass (F0={F0:.2f})" # Label F0 correctly
    plt.title(f"Predicted Yield Contour Plot ({biomass_type_label})", fontsize=14, pad=15)
    cbar = plt.colorbar(contour, ticks=np.linspace(0, 1, 11)); cbar.set_label(f"Predicted Yield (1 - F)", fontsize=12); cbar.ax.tick_params(labelsize=10)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.grid(True, linestyle=':', alpha=0.5)
    f0_str = f"{F0:.2f}".replace('.', 'p'); filename = f"yield_contour_plot_F0_{f0_str}.png"
    output_path = output_dir / filename; output_path.parent.mkdir(parents=True, exist_ok=True)
    try: plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(f"  Saved contour plot to {output_path}")
    except Exception as e: print(f"  Error saving contour plot: {e}")
    plt.close()


# --- Plot 4: Observed vs Dose Overview (Manual Layout) ---
# (Using the finalized version from previous steps)
def plot_overview_observed_vs_dose(df_raw_subset: pd.DataFrame, best_k: float, best_alpha: float, output_dir: Path, config_name: str):
    """ Plots ONLY OBSERVED data points vs. cumulative dose, using manual layout. """
    required_cols = ['experiment_id', 'observed_frac', 'total_passages_650', 'total_passages_1000',
                     'wash_procedure', 'biomass_type']
    if not all(col in df_raw_subset.columns for col in required_cols):
        missing = set(required_cols) - set(df_raw_subset.columns); print(f"Warning: Skipping plot_overview_observed_vs_dose missing: {missing}"); return

    # Corrected: Filter NA experiment_id before unique()
    df_plot_data = df_raw_subset.dropna(subset=['experiment_id']).copy()
    if df_plot_data.empty: print(f"Warning: No experiments after dropping NA experiment_id for plot_overview_observed_vs_dose ({config_name})."); return
    experiment_ids = sorted(df_plot_data['experiment_id'].unique())
    n_total = len(experiment_ids)
    if n_total == 0: print("Warning: No experiments found for plot_overview_observed_vs_dose."); return
    if n_total > 7: print(f"Warning: plot_overview_observed_vs_dose layout (3+4) fixed for 7. Found {n_total}. Plotting first 7."); experiment_ids = experiment_ids[:7]; n_total = 7
    elif n_total < 7: print(f"Warning: plot_overview_observed_vs_dose layout fixed for 7. Found {n_total}. Layout might look sparse.")

    subplot_width=2; subplot_height=2; h_spacing=0.5; v_spacing=1.5; left_margin=1; right_margin=1; bottom_margin=1; top_margin=1.5
    n_top=3; n_bottom=4; has_top_row = n_total >= 1; has_bottom_row = n_total > 3
    total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing; total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
    fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
    if has_top_row and has_bottom_row: fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
    elif has_top_row or has_bottom_row: fig_height = top_margin + subplot_height + bottom_margin
    else: fig_height = top_margin + bottom_margin
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    bottom_y = bottom_margin / fig_height; bottom_h = subplot_height / fig_height
    top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
    top_h = subplot_height / fig_height
    top_row_content_width = fig_width - left_margin - right_margin; bottom_row_content_width = fig_width - left_margin - right_margin
    top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
    bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
    axes_top = []; axes_bottom = []
    if has_top_row:
        for i in range(n_top): ax_left = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_top.append(fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))
    if has_bottom_row:
        for i in range(n_bottom): ax_left = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_bottom.append(fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))
    all_axes = axes_top + axes_bottom
    top_experiments = experiment_ids[:min(n_top, n_total)]; bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]

    for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)):
        # Use df_plot_data which has NAs dropped
        exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
        if exp_data.empty: continue
        x_vals = compute_cumulative_dose_values(exp_data, best_k, best_alpha)
        ax.plot(x_vals, exp_data['observed_frac'], marker='o', linestyle='-')
        ax.set_xlabel('');
        if j == 0: ax.set_ylabel("Observed Fraction Intact", fontsize=10)
        else: ax.set_ylabel('')
        row0 = exp_data.iloc[0]; wash = row0['wash_procedure']; biomass = row0['biomass_type']
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
        ax.set_title(f'Experiment {exp_id}', fontsize=10); ax.set_ylim(0, 1.05); ax.grid(True, linestyle=':', alpha=0.6)

    for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)):
        exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
        if exp_data.empty: continue
        x_vals = compute_cumulative_dose_values(exp_data, best_k, best_alpha)
        ax.plot(x_vals, exp_data['observed_frac'], marker='o', linestyle='-')
        ax.set_xlabel('')
        if j == 0: ax.set_ylabel("Observed Fraction Intact", fontsize=10)
        else: ax.set_ylabel('')
        row0 = exp_data.iloc[0]; wash = row0['wash_procedure']; biomass = row0['biomass_type']
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
        ax.set_title(f'Experiment {exp_id}', fontsize=10); ax.set_ylim(0, 1.05); ax.grid(True, linestyle=':', alpha=0.6)

    for k in range(n_total, 7):
        if k < len(all_axes): all_axes[k].axis('off')
    fig.text(0.5, (bottom_margin * 0.4) / fig_height , "Cumulative Dose", ha='center', va='center', fontsize=12)
    fig.suptitle("Overview: Observed Fraction Intact vs. Cumulative Dose", fontsize=14, y=1.0 - (top_margin*0.3)/fig_height)

    output_filename = f"overview_observed_vs_dose.png"
    output_path = output_dir / output_filename; output_path.parent.mkdir(parents=True, exist_ok=True)
    try: fig.savefig(output_path, dpi=300); print(f"  Saved observed data overview plot to {output_path}")
    except Exception as e: print(f"  Error saving observed data overview plot: {e}")
    plt.close(fig)


# --- Plot 5: Observed vs Step Overview (Manual Layout & sns.lineplot) ---
# (Using the finalized version from previous steps)
def plot_overview_observed_vs_step(df_raw_subset: pd.DataFrame, output_dir: Path, config_name: str):
    """ Plots ONLY OBSERVED data points vs. process_step, using manual layout and sns.lineplot. """
    required_cols = ['experiment_id', 'process_step', 'observed_frac', 'wash_procedure', 'biomass_type']
    if not all(col in df_raw_subset.columns for col in required_cols):
        missing = set(required_cols) - set(df_raw_subset.columns); print(f"Warning: Skipping plot_overview_observed_vs_step missing: {missing}"); return

    # Corrected: Filter NA experiment_id before unique()
    df_plot_data = df_raw_subset.dropna(subset=['experiment_id']).copy()
    if df_plot_data.empty: print(f"Warning: No experiments after dropping NA experiment_id for plot_overview_observed_vs_step ({config_name})."); return
    experiment_ids = sorted(df_plot_data['experiment_id'].unique())
    n_total = len(experiment_ids)
    if n_total == 0: print("Warning: No experiments found for plot_overview_observed_vs_step."); return
    if n_total > 7: print(f"Warning: plot_overview_observed_vs_step layout (3+4) fixed for 7. Found {n_total}. Plotting first 7."); experiment_ids = experiment_ids[:7]; n_total = 7
    elif n_total < 7: print(f"Warning: plot_overview_observed_vs_step layout fixed for 7. Found {n_total}. Layout might look sparse.")

    subplot_width=2; subplot_height=2; h_spacing=0.5; v_spacing=1.5; left_margin=1; right_margin=1; bottom_margin=1; top_margin=1.5
    n_top=3; n_bottom=4; has_top_row = n_total >= 1; has_bottom_row = n_total > 3
    total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing; total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
    fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
    if has_top_row and has_bottom_row: fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
    elif has_top_row or has_bottom_row: fig_height = top_margin + subplot_height + bottom_margin
    else: fig_height = top_margin + bottom_margin
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    bottom_y = bottom_margin / fig_height; bottom_h = subplot_height / fig_height
    top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
    top_h = subplot_height / fig_height
    top_row_content_width = fig_width - left_margin - right_margin; bottom_row_content_width = fig_width - left_margin - right_margin
    top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
    bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
    axes_top = []; axes_bottom = []
    if has_top_row:
        for i in range(n_top): ax_left = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_top.append(fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))
    if has_bottom_row:
        for i in range(n_bottom): ax_left = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_bottom.append(fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))
    all_axes = axes_top + axes_bottom
    top_experiments = experiment_ids[:min(n_top, n_total)]; bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]

    for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)):
        exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
        if exp_data.empty: continue
        sns.lineplot(data=exp_data, x='process_step', y='observed_frac', marker='o', ax=ax, sort=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9); ax.set_xlabel('')
        if j == 0: ax.set_ylabel("Observed Fraction Intact", fontsize=10)
        else: ax.set_ylabel('')
        row0 = exp_data.iloc[0]; wash = row0['wash_procedure']; biomass = row0['biomass_type']
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
        ax.set_title(f'Experiment {exp_id}', fontsize=10); ax.set_ylim(0, 1.05); ax.grid(True, linestyle=':', alpha=0.6)

    for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)):
        exp_data = df_plot_data[df_plot_data['experiment_id'] == exp_id].sort_index()
        if exp_data.empty: continue
        sns.lineplot(data=exp_data, x='process_step', y='observed_frac', marker='o', ax=ax, sort=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9); ax.set_xlabel('')
        if j == 0: ax.set_ylabel("Observed Fraction Intact", fontsize=10)
        else: ax.set_ylabel('')
        row0 = exp_data.iloc[0]; wash = row0['wash_procedure']; biomass = row0['biomass_type']
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
        ax.set_title(f'Experiment {exp_id}', fontsize=10); ax.set_ylim(0, 1.05); ax.grid(True, linestyle=':', alpha=0.6)

    for k in range(n_total, 7):
        if k < len(all_axes): all_axes[k].axis('off')
    fig.text(0.5, (bottom_margin * 0.4) / fig_height , "Process Step", ha='center', va='center', fontsize=12)
    fig.suptitle("Overview: Observed Fraction Intact over Process Steps", fontsize=14, y=1.0 - (top_margin*0.3)/fig_height)

    output_filename = f"overview_observed_vs_process_step.png"
    output_path = output_dir / output_filename; output_path.parent.mkdir(parents=True, exist_ok=True)
    try: plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(f"  Saved observed data vs process step overview plot to {output_path}")
    except Exception as e: print(f"  Error saving observed data vs process step overview plot: {e}")
    plt.close(fig)


# --- Plot 6: DNA vs Process Step (Log Scale Y, Manual Layout) ---
# (Corrected version including NA filtering and mticker import)
def plot_dna_vs_step_matplotlib(df_dna_pred: pd.DataFrame, output_dir: Path, config_name: str):
    """ DNA vs process step overview using manual layout and log scale Y. """
    required_cols = ['experiment_id', 'process_step', 'dna_conc', 'dna_pred', 'wash_procedure', 'biomass_type']
    if not all(col in df_dna_pred.columns for col in required_cols):
        missing = set(required_cols) - set(df_dna_pred.columns); print(f"Warning: Skipping plot_dna_vs_step_matplotlib missing: {missing}"); return

    # Corrected: Filter NA experiment_id before unique()
    df_plot_data = df_dna_pred.dropna(subset=['experiment_id']).copy()
    if df_plot_data.empty: print(f"Warning: No experiments after dropping NA experiment_id for plot_dna_vs_step_matplotlib ({config_name})."); return
    experiment_ids = sorted(df_plot_data['experiment_id'].unique())
    n_total = len(experiment_ids)
    if n_total == 0: print("Warning: No experiments found for plot_dna_vs_step_matplotlib."); return
    if n_total > 7: print(f"Warning: plot_dna_vs_step layout (3+4) fixed for 7. Found {n_total}. Plotting first 7."); experiment_ids = experiment_ids[:7]; n_total = 7
    elif n_total < 7: print(f"Warning: plot_dna_vs_step layout fixed for 7. Found {n_total}. Layout might look sparse.")

    subplot_width = 2; subplot_height = 2; h_spacing = 0.5; v_spacing = 1.5; left_margin = 1; right_margin = 1; bottom_margin = 1; top_margin = 1.5
    n_top = 3; n_bottom = 4; has_top_row = n_total >= 1; has_bottom_row = n_total > 3
    total_top_width = n_top * subplot_width + (n_top - 1) * h_spacing; total_bottom_width = n_bottom * subplot_width + (n_bottom - 1) * h_spacing
    fig_width = max(total_top_width, total_bottom_width) + left_margin + right_margin
    if has_top_row and has_bottom_row: fig_height = top_margin + subplot_height + v_spacing + subplot_height + bottom_margin
    elif has_top_row or has_bottom_row: fig_height = top_margin + subplot_height + bottom_margin
    else: fig_height = top_margin + bottom_margin
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    bottom_y = bottom_margin / fig_height; bottom_h = subplot_height / fig_height
    top_y = (bottom_margin + subplot_height + v_spacing) / fig_height if has_bottom_row else bottom_y
    top_h = subplot_height / fig_height
    top_row_content_width = fig_width - left_margin - right_margin; bottom_row_content_width = fig_width - left_margin - right_margin
    top_left_offset = left_margin + (top_row_content_width - total_top_width) / 2
    bottom_left_offset = left_margin + (bottom_row_content_width - total_bottom_width) / 2
    axes_top = []; axes_bottom = []
    if has_top_row:
        for i in range(n_top): ax_left = (top_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_top.append(fig.add_axes([ax_left, top_y, subplot_width / fig_width, top_h]))
    if has_bottom_row:
        for i in range(n_bottom): ax_left = (bottom_left_offset + i * (subplot_width + h_spacing)) / fig_width; axes_bottom.append(fig.add_axes([ax_left, bottom_y, subplot_width / fig_width, bottom_h]))
    all_axes = axes_top + axes_bottom
    top_experiments = experiment_ids[:min(n_top, n_total)]; bottom_experiments = experiment_ids[n_top:min(n_top + n_bottom, n_total)]
    obs_color = 'blue'; pred_color = 'red'; min_yaxis_val = np.inf; max_yaxis_val = -np.inf

    def plot_single_dna_step(ax, exp_id_single, df_plot):
        nonlocal min_yaxis_val, max_yaxis_val
        exp_data = df_plot[df_plot['experiment_id'] == exp_id_single].sort_index()
        if exp_data.empty: return
        obs_plot = exp_data[exp_data['dna_conc'] > 0]; pred_plot = exp_data[exp_data['dna_pred'] > 0]
        if not obs_plot.empty: min_yaxis_val = min(min_yaxis_val, obs_plot['dna_conc'].min()); max_yaxis_val = max(max_yaxis_val, obs_plot['dna_conc'].max())
        if not pred_plot.empty: min_yaxis_val = min(min_yaxis_val, pred_plot['dna_pred'].min()); max_yaxis_val = max(max_yaxis_val, pred_plot['dna_pred'].max())
        sns.lineplot(data=exp_data, x='process_step', y='dna_conc', marker='o', ax=ax, label='Observed', color=obs_color, sort=False, legend=False)
        sns.lineplot(data=exp_data, x='process_step', y='dna_pred', marker='x', linestyle='--', ax=ax, label='Predicted', color=pred_color, sort=False, legend=False)
        ax.set_yscale('log'); ax.set_xlabel(''); ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9); plt.setp(ax.get_yticklabels(), fontsize=9)
        row0 = exp_data.iloc[0]; wash = row0['wash_procedure']; biomass = row0['biomass_type']
        ax.text(0.95, 0.95, f"{wash}\n{biomass}", transform=ax.transAxes, va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
        ax.set_title(f'Experiment {exp_id_single}', fontsize=10); ax.grid(True, which='major', linestyle='-', alpha=0.6); ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    for j, (ax, exp_id) in enumerate(zip(axes_top, top_experiments)): plot_single_dna_step(ax, exp_id, df_plot_data); plt.setp(ax.get_yticklabels(), visible=(j==0))
    for j, (ax, exp_id) in enumerate(zip(axes_bottom, bottom_experiments)): plot_single_dna_step(ax, exp_id, df_plot_data); plt.setp(ax.get_yticklabels(), visible=(j==0)); plt.setp(ax.get_xticklabels(), visible=False)

    if np.isfinite(min_yaxis_val) and np.isfinite(max_yaxis_val) and max_yaxis_val > min_yaxis_val:
        y_min_log = np.floor(np.log10(min_yaxis_val)); y_max_log = np.ceil(np.log10(max_yaxis_val))
        for ax in all_axes:
             if ax.has_data():
                 ax.set_ylim(10**y_min_log, 10**y_max_log)
                 ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False)) # Corrected formatter
                 ax.yaxis.set_minor_formatter(mticker.NullFormatter()) # Hide minor ticks labels if desired
    if axes_top and axes_top[0].has_data(): axes_top[0].set_ylabel("DNA [ng/µL] (Log)", fontsize=10)
    if axes_bottom and axes_bottom[0].has_data(): axes_bottom[0].set_ylabel("DNA [ng/µL] (Log)", fontsize=10)
    for k in range(n_total, 7):
        if k < len(all_axes): all_axes[k].axis('off')
    fig.text(0.5, (bottom_margin * 0.4) / fig_height , "Process Step", ha='center', va='center', fontsize=12)
    fig.suptitle("Overview: DNA Concentration vs. Process Step", fontsize=14, y=1.0 - (top_margin*0.3)/fig_height)
    obs_handle = Line2D([], [], marker='o', color=obs_color, linestyle='-', label='Observed'); pred_handle = Line2D([], [], marker='x', color=pred_color, linestyle='--', label='Predicted')
    fig.legend(handles=[obs_handle, pred_handle], loc='upper center', bbox_to_anchor=(0.5, 1 - (top_margin / fig_height) * 0.15), ncol=2, fontsize=10)
    output_filename = f"overview_dna_vs_process_step.png"
    output_path = output_dir / output_filename; output_path.parent.mkdir(parents=True, exist_ok=True)
    try: plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(f"  Saved DNA vs process step overview plot to {output_path}")
    except Exception as e: print(f"  Error saving DNA vs process step overview plot: {e}")
    plt.close(fig)


# --- Plot 7: DNA Parity Plot (Log-Log Scale, Matplotlib/Seaborn) ---
# (Corrected version including NA filtering and mticker import)
def plot_dna_parity_matplotlib(df_dna_pred: pd.DataFrame, output_dir: Path, config_name: str):
    """ DNA parity plot using Matplotlib/Seaborn on a Log-Log scale. """
    required_cols = ['dna_conc', 'dna_pred', 'wash_procedure', 'biomass_type', 'process_step']
    if not all(col in df_dna_pred.columns for col in required_cols):
        missing = set(required_cols) - set(df_dna_pred.columns); print(f"Warning: Skipping plot_dna_parity_matplotlib missing: {missing}"); return

    plot_data_base = df_dna_pred.dropna(subset=['dna_conc', 'dna_pred']).copy()
    plot_data_base = plot_data_base[plot_data_base['process_step'].str.lower() != "resuspended biomass"]
    plot_data = plot_data_base[(plot_data_base['dna_conc'] > 0) & (plot_data_base['dna_pred'] > 0)].copy()
    if plot_data.empty: print(f"Warning: No valid positive data points found for DNA parity plot ({config_name}). Skipping."); return

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    markers = {"fresh biomass": "o", "frozen biomass": "s"}
    wash_types = sorted(plot_data["wash_procedure"].unique())
    palette = sns.color_palette("Set2", n_colors=len(wash_types))
    plot_data['Wash Type'] = plot_data['wash_procedure'].str.replace(" wash", "")
    plot_data['Biomass'] = plot_data['biomass_type'].str.replace(" biomass", "").str.capitalize()

    g = sns.scatterplot(data=plot_data, x="dna_conc", y="dna_pred", hue="Wash Type", style="Biomass",
                        markers={"Fresh": "o", "Frozen": "s"}, palette=palette, s=50, edgecolor="k", alpha=0.8, legend="full")
    g.set(xscale="log", yscale="log")
    min_val = min(plot_data["dna_conc"].min(), plot_data["dna_pred"].min()) * 0.8
    max_val = max(plot_data["dna_conc"].max(), plot_data["dna_pred"].max()) * 1.2
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, zorder=1, label="y = x")
    plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
    plt.xlabel("Observed DNA [ng/µL] (Log Scale)", fontsize=11); plt.ylabel("Predicted DNA [ng/µL] (Log Scale)", fontsize=11)
    plt.title(f"DNA Parity Plot ({config_name.replace('_',' ').title()})", fontsize=13, pad=15)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9)
    g.xaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False)) # Corrected formatter
    g.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
    g.xaxis.set_minor_formatter(mticker.NullFormatter()) # Hide minor ticks
    g.yaxis.set_minor_formatter(mticker.NullFormatter())
    plt.legend(title="Condition", title_fontsize=10, fontsize=9, bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.grid(True, which='major', linestyle='-', alpha=0.6); plt.grid(True, which='minor', linestyle=':', alpha=0.3)

    output_filename = f"parity_plot_dna_loglog.png"
    output_path = output_dir / output_filename; output_path.parent.mkdir(parents=True, exist_ok=True)
    try: plt.savefig(output_path, dpi=300, bbox_inches='tight'); print(f"  Saved DNA parity plot to {output_path}")
    except Exception as e: print(f"  Error saving DNA parity plot: {e}")
    plt.close()