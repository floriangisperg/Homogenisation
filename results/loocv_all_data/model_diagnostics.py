import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# RESULTS_DIR = Path("results") # Adjust if your script is not in the root
# LOOCV_DIR = RESULTS_DIR / "loocv_all_data"
LOOCV_PREDS_FILE = "loocv_predictions.csv"
DIAGNOSTICS_OUTPUT_DIR = Path("diagnostics")
DIAGNOSTICS_OUTPUT_DIR.mkdir(exist_ok=True)

# Set plot style
# plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")

# --- Load Data ---
try:
    loocv_preds_df = pd.read_csv(LOOCV_PREDS_FILE)
    print(f"Loaded LOOCV predictions from: {LOOCV_PREDS_FILE}")
except FileNotFoundError:
    print(f"ERROR: LOOCV predictions file not found at {LOOCV_PREDS_FILE}")
    exit()

# --- Data Preparation for Analysis ---
# Calculate DNA residuals (Observed - Predicted)
loocv_preds_df['dna_residual'] = loocv_preds_df['dna_conc'] - loocv_preds_df['dna_pred']
loocv_preds_df['dna_abs_residual'] = loocv_preds_df['dna_residual'].abs()

# Calculate Absolute Percentage Error (handle zeros in observed)
obs = loocv_preds_df['dna_conc']
pred = loocv_preds_df['dna_pred']
# Use a small epsilon to avoid division by zero and issues with very small observed values
epsilon = 1e-6
loocv_preds_df['dna_ape'] = np.where(obs > epsilon, np.abs((obs - pred) / obs) * 100, np.nan)

# Filter out rows where prediction wasn't possible or observed is missing
analysis_df = loocv_preds_df.dropna(subset=['dna_conc', 'dna_pred', 'dna_residual']).copy()

# Also filter out the 'resuspended biomass' step for error analysis
analysis_df = analysis_df[analysis_df['process_step'].str.lower() != 'resuspended biomass'].copy()

# Add log versions for plotting if needed (handle non-positive values)
analysis_df['log10_dna_conc'] = np.log10(analysis_df['dna_conc'].clip(lower=epsilon))
analysis_df['log10_dna_pred'] = np.log10(analysis_df['dna_pred'].clip(lower=epsilon))
analysis_df['log10_delta_F'] = np.log10(analysis_df['delta_F'].clip(lower=epsilon)) # May need adjustment if delta_F can be zero


print(f"Analyzing {len(analysis_df)} points with valid LOOCV DNA predictions.")
if analysis_df.empty:
    print("No valid data points for diagnostic analysis. Exiting.")
    exit()

# --- Generate Diagnostic Plots ---

# 1. Residuals vs. Predicted (Linear Scale)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=analysis_df, x='dna_pred', y='dna_residual', hue='biomass_type', style='wash_procedure', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', lw=1)
plt.title('LOOCV DNA: Residuals vs. Predicted Value')
plt.xlabel('Predicted DNA [ng/µL]')
plt.ylabel('Residual (Observed - Predicted)')
plt.grid(True, linestyle=':')
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_vs_pred.png", bbox_inches='tight')
plt.show()

# 2. Residuals vs. Predicted (Log Scale - if values span orders of magnitude)
# Check if log scales are appropriate
if analysis_df['dna_pred'].min() > epsilon and analysis_df['dna_pred'].max() / analysis_df['dna_pred'].min() > 100:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=analysis_df, x='log10_dna_pred', y='dna_residual', hue='biomass_type', style='wash_procedure', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', lw=1)
    plt.title('LOOCV DNA: Residuals vs. Log Predicted Value')
    plt.xlabel('Log10 Predicted DNA')
    plt.ylabel('Residual (Observed - Predicted)')
    plt.grid(True, linestyle=':')
    plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_vs_logpred.png", bbox_inches='tight')
    plt.show()

# 3. Residuals vs. Delta F (Lysis Fraction)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=analysis_df, x='delta_F', y='dna_residual', hue='biomass_type', style='wash_procedure', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', lw=1)
plt.title('LOOCV DNA: Residuals vs. Delta F (Lysis Fraction)')
plt.xlabel('Delta F (Fraction Lysed in Step)')
plt.ylabel('Residual (Observed - Predicted)')
plt.grid(True, linestyle=':')
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_vs_deltaF.png", bbox_inches='tight')
plt.show()

# 4. Residuals vs. F_before (Intact Fraction Before Step)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=analysis_df, x='F_before', y='dna_residual', hue='biomass_type', style='wash_procedure', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', lw=1)
plt.title('LOOCV DNA: Residuals vs. F_before (Fraction Intact Before Step)')
plt.xlabel('F_before')
plt.ylabel('Residual (Observed - Predicted)')
plt.grid(True, linestyle=':')
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_vs_Fbefore.png", bbox_inches='tight')
plt.show()

# 5. Residuals by Experiment ID (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_df, x='experiment_id', y='dna_residual', palette="coolwarm")
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.title('LOOCV DNA: Residuals by Experiment ID')
plt.xlabel('Experiment ID')
plt.ylabel('Residual (Observed - Predicted)')
plt.xticks(rotation=0)
plt.grid(True, axis='y', linestyle=':')
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_by_experiment.png", bbox_inches='tight')
plt.show()

# 6. Residuals by Process Step (Box Plot)
plt.figure(figsize=(12, 7))
# Order steps logically if possible (requires defining an order)
step_order = sorted(analysis_df['process_step'].unique()) # Basic alphabetical sort
try:
    # Attempt a more logical sort if step names allow
    def sort_key(step):
        step_low = step.lower()
        if "initial lysis" in step_low: return 0
        if "1st wash" in step_low: return 1
        if "2nd wash" in step_low: return 2
        if "3rd wash" in step_low: return 3
        if "4th wash" in step_low: return 4
        # Add other steps...
        return 100 # Default for others
    step_order = sorted(analysis_df['process_step'].unique(), key=sort_key)
except Exception:
    pass # Stick to alphabetical if sorting fails

sns.boxplot(data=analysis_df, x='process_step', y='dna_residual', order=step_order, palette="viridis")
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.title('LOOCV DNA: Residuals by Process Step')
plt.xlabel('Process Step')
plt.ylabel('Residual (Observed - Predicted)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_resid_by_step.png", bbox_inches='tight')
plt.show()

# 7. Absolute Percentage Error (APE) by Experiment ID
plt.figure(figsize=(10, 6))
sns.boxplot(data=analysis_df, x='experiment_id', y='dna_ape', palette="coolwarm")
plt.title('LOOCV DNA: Absolute Percentage Error (%) by Experiment ID')
plt.xlabel('Experiment ID')
plt.ylabel('Absolute Percentage Error (%)')
plt.xticks(rotation=0)
plt.ylim(bottom=0) # APE cannot be negative
plt.grid(True, axis='y', linestyle=':')
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_ape_by_experiment.png", bbox_inches='tight')
plt.show()

# 8. Absolute Percentage Error (APE) by Process Step
plt.figure(figsize=(12, 7))
sns.boxplot(data=analysis_df, x='process_step', y='dna_ape', order=step_order, palette="viridis")
plt.title('LOOCV DNA: Absolute Percentage Error (%) by Process Step')
plt.xlabel('Process Step')
plt.ylabel('Absolute Percentage Error (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(bottom=0) # APE cannot be negative
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_ape_by_step.png", bbox_inches='tight')
plt.show()


# 9. Predicted vs. Observed Faceted by Condition (Example: Biomass Type)
# Use log scale if data spans orders of magnitude and is positive
use_log_parity = analysis_df['dna_conc'].min() > epsilon and analysis_df['dna_pred'].min() > epsilon and analysis_df['dna_conc'].max() / analysis_df['dna_conc'].min() > 100

min_val = analysis_df[['dna_conc', 'dna_pred']].min().min() * 0.8
max_val = analysis_df[['dna_conc', 'dna_pred']].max().max() * 1.2
min_val = max(min_val, epsilon) # Ensure min_val is positive for log scale

g = sns.relplot(data=analysis_df, x='dna_conc', y='dna_pred',
                col='biomass_type', row='wash_procedure', # Facet by conditions
                hue='experiment_id', # Color points by experiment
                kind='scatter', palette='tab10', s=50, alpha=0.8, legend=False) # Removed legend for clarity

# Add y=x line to each facet
for ax in g.axes.flat:
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, zorder=1)
    if use_log_parity:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
        ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("Observed DNA (Log)")
        ax.set_ylabel("Predicted DNA (Log)")
    else:
        ax.set_xlabel("Observed DNA")
        ax.set_ylabel("Predicted DNA")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle=':')

g.fig.suptitle('LOOCV DNA Parity Plot (Faceted)', y=1.03)
plt.savefig(DIAGNOSTICS_OUTPUT_DIR / "dna_parity_faceted.png", bbox_inches='tight')
plt.show()


print(f"Diagnostic plots saved to: {DIAGNOSTICS_OUTPUT_DIR}")
