#!/usr/bin/env python3
"""
Run Cybench analysis section from analysis.ipynb
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description="Run Cybench analysis section from analysis.ipynb")
parser.add_argument(
    "--params-path",
    type=Path,
    default=Path("/home/lfy/BRIDGE/params/all_a_pyirt.csv"),
    help="Path to the IRT parameters CSV (default: %(default)s)",
)
parser.add_argument(
    "--baseline-params-path",
    type=Path,
    default=Path("/home/lfy/BRIDGE/params/all_a_pyirt_baseline.csv"),
    help="Path to the baseline parameters CSV (default: %(default)s)",
)
args = parser.parse_args()

# Setup paths
BASE_DIR = Path('/home/lfy/BRIDGE')
cybench_results_path = BASE_DIR / 'data' / 'cybench_normalized_results.jsonl'
params_path = args.params_path
baseline_params_path = args.baseline_params_path
all_runs_path = BASE_DIR / 'data' / 'all_runs.jsonl'
plots_dir = BASE_DIR / 'plots'
plots_dir.mkdir(exist_ok=True)

# Load Cybench task IDs, difficulty labels, and eval modes
def load_jsonl_records(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records

cybench_records = load_jsonl_records(cybench_results_path)
cybench_task_ids = {record['task_id'] for record in cybench_records}

# Create task_id -> difficulty_label mapping
task_difficulty_map = {}
task_eval_mode_map = {}
for record in cybench_records:
    task_id = record['task_id']
    if task_id not in task_difficulty_map:
        task_difficulty_map[task_id] = record.get('difficulty_label', 'unlabelled')
        task_eval_mode_map[task_id] = record.get('eval_mode', 'unguided')

print(f"# of task instances in Cybench: {len(cybench_task_ids)}")

# Count by difficulty
difficulty_counts = {}
for label in task_difficulty_map.values():
    difficulty_counts[label] = difficulty_counts.get(label, 0) + 1
print(f"Difficulty distribution: {difficulty_counts}")

# Count by eval_mode
eval_mode_counts = {}
for mode in task_eval_mode_map.values():
    eval_mode_counts[mode] = eval_mode_counts.get(mode, 0) + 1
print(f"Eval mode distribution: {eval_mode_counts}")

# Load IRT parameters
df = pd.read_csv(params_path)
# Handle unnamed first column for task_id
if df.columns[0] != 'task_id':
    df.rename(columns={df.columns[0]: 'task_id'}, inplace=True)

# Load baseline parameters
baseline_df = pd.read_csv(baseline_params_path)
if baseline_df.columns[0] != 'task_id':
    baseline_df.rename(columns={baseline_df.columns[0]: 'task_id'}, inplace=True)

# Merge baseline into main df
df = df.merge(baseline_df[['task_id', 'baseline_difficulty_logit', 'success_rate']],
              on='task_id', how='left')

# Load task sources from all_runs.jsonl for METR data
metr_records = load_jsonl_records(all_runs_path)
metr_task_sources = {}
for record in metr_records:
    task_id = record.get('task_id')
    task_source = record.get('task_source')
    if task_id and task_source and task_id not in metr_task_sources:
        metr_task_sources[task_id] = task_source.lower().replace('-', '')

# Mark task sources
df['task_source'] = df['task_id'].map(metr_task_sources)
df.loc[df['task_id'].isin(cybench_task_ids), 'task_source'] = 'cybench'
df['task_source'] = df['task_source'].fillna('other')

# Add difficulty_label and eval_mode columns
df['difficulty_label'] = df['task_id'].map(task_difficulty_map).fillna('unlabelled')
df['eval_mode'] = df['task_id'].map(task_eval_mode_map).fillna('unguided')

# Fit b vs log(human_minutes) regression on METR-only data
# This replicates the logic from analysis.ipynb before the Cybench section
METR_SOURCES = {'hcast', 'rebench', 'swaa'}
metr_fit_df = df.dropna(subset=['b', 'human_minutes']).copy()
metr_fit_df = metr_fit_df[np.isfinite(metr_fit_df['b']) & np.isfinite(metr_fit_df['human_minutes'])]
metr_fit_df = metr_fit_df[metr_fit_df['human_minutes'] > 0]
metr_fit_df = metr_fit_df[metr_fit_df['task_source'].isin(METR_SOURCES)]

if len(metr_fit_df) < 10:
    raise ValueError(f"Not enough METR data for regression fit: only {len(metr_fit_df)} rows")

x_b = metr_fit_df['b'].to_numpy()
y_log_minutes = np.log(metr_fit_df['human_minutes'].to_numpy())

reg = stats.linregress(x_b, y_log_minutes)
slope = reg.slope
intercept = reg.intercept
r_squared = reg.rvalue ** 2

print(f"\nLinear Regression: b vs. log(minutes) coefficients (METR-only, n={len(metr_fit_df)}):")
print(f"  slope: {slope:.6f}")
print(f"  intercept: {intercept:.6f}")
print(f"  R²: {r_squared:.4f}")

def predict_minutes_from_b(b_values):
    """Predict human minutes from IRT difficulty parameter b"""
    return np.exp(slope * b_values + intercept)

# Fit baseline prediction model using METR data (non-Cybench tasks with human_minutes)
metr_mask = (df['task_source'].isin(METR_SOURCES)) & (df['human_minutes'].notna()) & (df['baseline_difficulty_logit'].notna())
metr_data = df[metr_mask]

if len(metr_data) > 10:
    # Fit log(human_minutes) ~ baseline_difficulty_logit
    baseline_slope, baseline_intercept, r_value, p_value, std_err = stats.linregress(
        metr_data['baseline_difficulty_logit'],
        np.log(metr_data['human_minutes'])
    )
    print(f"\nBaseline model fit (on METR data, n={len(metr_data)}):")
    print(f"  log(minutes) = {baseline_slope:.4f} * baseline_difficulty + {baseline_intercept:.4f}")
    print(f"  R² = {r_value**2:.3f}")
else:
    # Fallback: use same slope but adjust intercept for scale difference
    baseline_slope = slope
    baseline_intercept = intercept
    print("\nWarning: Not enough METR data for baseline fit, using IRT coefficients")

def predict_minutes_from_baseline(baseline_diff_values):
    """Predict human minutes from baseline difficulty (logit)"""
    return np.exp(baseline_slope * baseline_diff_values + baseline_intercept)


def normalize_task_id(task_id: str, *, strip_difficulty=False, append_suffix=None):
    """Normalize task_id strings so paths match across datasets.

    Args:
        task_id: Raw task identifier.
        strip_difficulty: If True, remove difficulty tokens (easy/very_easy/medium/hard).
        append_suffix: Optional suffix to append (e.g., 'unguided').
    """
    if not isinstance(task_id, str):
        return task_id
    normalized = task_id.strip().lower()
    normalized = normalized.replace('/', '_').replace(' ', '_').replace('-', '_')
    if strip_difficulty:
        difficulty_tokens = ['very_easy', 'easy', 'medium', 'hard']
        for token in difficulty_tokens:
            normalized = normalized.replace(f"_{token}_", "_")
            if normalized.endswith(f"_{token}"):
                normalized = normalized[: -len(token) - 1]
            if normalized.startswith(f"{token}_"):
                normalized = normalized[len(token) + 1 :]
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    if append_suffix and not normalized.endswith(f"_{append_suffix}"):
        normalized = f"{normalized}_{append_suffix}"
    return normalized

# Get Cybench predictions
cybench_predictions = df[(df['task_id'].isin(cybench_task_ids)) &
                         (df['task_source'] == 'cybench') &
                         (df['b'].notna())].copy()

print(f"\nCybench task instances with difficulty estimates: {len(cybench_predictions)}")

# Predict human time (IRT)
cybench_predictions['predicted_minutes'] = predict_minutes_from_b(cybench_predictions['b'])

# Predict human time (Baseline)
cybench_predictions['predicted_minutes_baseline'] = predict_minutes_from_baseline(
    cybench_predictions['baseline_difficulty_logit']
)

# Filter to tasks with actual human_minutes for evaluation
mask = cybench_predictions['human_minutes'].notna()
print(f"Cybench task instances with actual FST annotations: {mask.sum()}")

if mask.sum() > 0:
    # Surface the exact data points used in the first subplot
    plot_points = (
        cybench_predictions.loc[mask, ['task_id', 'difficulty_label', 'eval_mode',
                                       'human_minutes', 'predicted_minutes']]
        .copy()
    )
    plot_points['human_minutes'] = plot_points['human_minutes'].round(2)
    plot_points['predicted_minutes'] = plot_points['predicted_minutes'].round(2)
    plot_points.sort_values('human_minutes', inplace=True)
    print("\nData used for Predicted vs Actual subplot (sorted by actual FST):")
    print(plot_points.to_string(index=False))

    # Calculate metrics for IRT
    actual = cybench_predictions.loc[mask, 'human_minutes'].values
    predicted = cybench_predictions.loc[mask, 'predicted_minutes'].values
    predicted_baseline = cybench_predictions.loc[mask, 'predicted_minutes_baseline'].values

    # R² on log scale (IRT)
    # y_actual = np.log(actual)
    # y_pred = np.log(predicted)
    # print(y_actual)
    # print(y_pred)
    y_actual = actual
    y_pred = predicted
    try:
        print("SCIKIT", r2_score(y_actual, y_pred))
    except:
        pass
    try:
        print("PEARSONR", stats.pearsonr(y_actual, y_pred))
    except:
        pass
    try:
        print("SPEARMANR", stats.spearmanr(y_actual, y_pred))
    except:
        pass
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r_squared_cybench = 1 - (ss_res / ss_tot)

    # R² on log scale (Baseline)
    y_pred_baseline = np.log(predicted_baseline)
    ss_res_baseline = np.sum((y_actual - y_pred_baseline) ** 2)
    r_squared_baseline = 1 - (ss_res_baseline / ss_tot)

    # MAE on log scale
    mae_log = np.mean(np.abs(y_actual - y_pred))
    mae_log_baseline = np.mean(np.abs(y_actual - y_pred_baseline))

    # Median error ratio
    error_ratio = predicted / actual
    median_error_ratio = np.median(error_ratio)
    error_ratio_baseline = predicted_baseline / actual
    median_error_ratio_baseline = np.median(error_ratio_baseline)

    # Within 2x
    within_2x = np.mean((error_ratio >= 0.5) & (error_ratio <= 2.0)) * 100
    within_2x_baseline = np.mean((error_ratio_baseline >= 0.5) & (error_ratio_baseline <= 2.0)) * 100

    print(f"\nCybench Prediction Metrics (n={mask.sum()}):")
    print(f"{'Metric':<25} {'IRT':>12} {'Baseline':>12}")
    print(f"{'-'*50}")
    print(f"{'R² (log scale, cybench)':<25} {r_squared_cybench:>12.3f} {r_squared_baseline:>12.3f}")
    print(f"{'MAE (log scale)':<25} {mae_log:>12.3f} {mae_log_baseline:>12.3f}")
    print(f"{'Median error ratio':<25} {median_error_ratio:>11.2f}x {median_error_ratio_baseline:>11.2f}x")
    print(f"{'Within 2x (%, cybench)':<25} {within_2x:>12.1f} {within_2x_baseline:>12.1f}")

    # Create log error columns for visualization
    cybench_predictions.loc[mask, 'log_error'] = y_pred - y_actual
    cybench_predictions.loc[mask, 'log_error_baseline'] = y_pred_baseline - y_actual

    # Define colors for difficulty levels
    difficulty_colors = {
        'very_easy': '#2ecc71',  # Green
        'easy': '#3498db',       # Blue
        'medium': '#f39c12',     # Orange
        'hard': '#e74c3c',       # Red
        'unlabelled': '#95a5a6'  # Gray
    }

    difficulty_order = ['very_easy', 'easy', 'medium', 'hard', 'unlabelled']
    difficulty_labels_display = {
        'very_easy': 'Very Easy',
        'easy': 'Easy',
        'medium': 'Medium',
        'hard': 'Hard',
        'unlabelled': 'Unlabelled'
    }

    # Define markers for eval modes
    eval_mode_markers = {
        'unguided': 'o',  # Circle
        'guided': 's',    # Square
    }

    # Check what eval modes are present
    eval_modes_present = cybench_predictions.loc[mask, 'eval_mode'].unique()
    print(f"\nEval modes in data: {list(eval_modes_present)}")

    plot_df = cybench_predictions.loc[mask].copy()

    def register_handle(handle, label, legend_handles, legend_labels):
        """Register a legend handle once so all subplots share one legend."""
        if label not in legend_labels:
            legend_labels.append(label)
            legend_handles.append(handle)

    def compute_shared_limits(series_list):
        """Compute shared min/max across provided series for consistent axes."""
        valid_series = []
        for series in series_list:
            if series is None:
                continue
            s = pd.to_numeric(series, errors='coerce')
            s = s[np.isfinite(s) & (s > 0)]
            if not s.empty:
                valid_series.append(s)
        if not valid_series:
            raise ValueError("No valid values available to compute plot limits.")

        min_val = min(s.min() for s in valid_series)
        max_val = max(s.max() for s in valid_series)
        # Apply a small padding to avoid points sitting on the boundary
        padding = 1.15
        return min_val / padding, max_val * padding

    def plot_prediction_vs_actual(ax, plot_subset, prediction_col, model_label,
                                  r_squared_value, within_2x_pct, shared_limits,
                                  legend_handles, legend_labels):
        """Scatter plot for predicted vs actual FST for a single model using shared axes."""
        if plot_subset.empty:
            print(f"\nNo data available to plot for {model_label}.")
            return

        min_val, max_val = shared_limits

        for diff in difficulty_order:
            for mode in ['unguided', 'guided']:
                diff_mode_mask = (plot_subset['difficulty_label'] == diff) & (plot_subset['eval_mode'] == mode)
                if diff_mode_mask.sum() > 0:
                    marker = eval_mode_markers.get(mode, 'o')
                    scatter = ax.scatter(
                        plot_subset.loc[diff_mode_mask, 'human_minutes'],
                        plot_subset.loc[diff_mode_mask, prediction_col],
                        c=difficulty_colors[diff],
                        marker=marker,
                        label=f"{difficulty_labels_display[diff]} ({mode[0].upper()})",
                        alpha=0.75,
                        s=60,
                        edgecolors='white',
                        linewidths=0.5
                    )
                    register_handle(scatter, scatter.get_label(), legend_handles, legend_labels)

        line, = ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        register_handle(line, line.get_label(), legend_handles, legend_labels)

        band = ax.fill_between([min_val, max_val], [min_val/2, max_val/2],
                               [min_val*2, max_val*2], alpha=0.15, color='gray',
                               label='2x Error Band')
        register_handle(band, band.get_label(), legend_handles, legend_labels)

        ax.set_title(
            f'Cybench: Predicted vs Actual FST ({model_label})\nR²={r_squared_value:.3f}, within 2x={within_2x_pct:.1f}% (n={len(plot_subset)})',
            fontsize=13, fontweight='bold'
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3)

    # ==================== Gemini Estimation Alignment Data ====================
    gemini_estimations_path = BASE_DIR / 'data' / 'cybench_time_estimations_google_gemini-3-pro-preview.jsonl'
    gemini_records = load_jsonl_records(gemini_estimations_path)
    gemini_df = pd.DataFrame(gemini_records)
    alignment_df = pd.DataFrame()
    within_2x_gemini = float('nan')
    r_squared_gemini = float('nan')

    if not gemini_df.empty and 'task_id' in gemini_df.columns:
        expected_cols = ['task_id', 'fst_minutes', 'estimated_minutes']
        missing_cols = [col for col in expected_cols if col not in gemini_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in Gemini estimations file: {missing_cols}")

        plot_df['task_id_norm'] = plot_df['task_id'].apply(normalize_task_id)
        gemini_df['task_id_norm'] = gemini_df['task_id'].apply(
            lambda t: normalize_task_id(t, strip_difficulty=True, append_suffix='unguided')
        )

        plot_task_ids = set(plot_df['task_id_norm'].dropna())
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', None)
        # print(gemini_df['task_id_norm'])
        # print(plot_task_ids)
        gemini_df = gemini_df[['task_id', 'task_id_norm', 'fst_minutes', 'estimated_minutes']]
        gemini_df = gemini_df[gemini_df['task_id_norm'].isin(plot_task_ids)]
        non_gemini_df = gemini_df[~gemini_df['task_id_norm'].isin(plot_task_ids)]

        alignment_df = plot_df[['task_id', 'task_id_norm', 'human_minutes']].merge(
            gemini_df, on='task_id_norm', how='inner', suffixes=('_irt', '_gemini')
        )
        alignment_df = alignment_df.dropna(subset=['estimated_minutes', 'human_minutes'])
        alignment_df = alignment_df[(alignment_df['estimated_minutes'] > 0) & (alignment_df['human_minutes'] > 0)]

        print(f"\nGemini estimations matched to plotted tasks: {len(alignment_df)}")

        if len(alignment_df) > 0:
            error_ratio_gemini = alignment_df['estimated_minutes'].to_numpy() / alignment_df['human_minutes'].to_numpy()
            within_2x_gemini = np.mean((error_ratio_gemini >= 0.5) & (error_ratio_gemini <= 2.0)) * 100
            try:
                r_squared_gemini = r2_score(alignment_df['human_minutes'], alignment_df['estimated_minutes'])
            except Exception:
                r_squared_gemini = float('nan')
        else:
            print("No overlapping Gemini estimations for the plotted Cybench tasks.")
    else:
        print("Gemini estimations file is empty or missing task_id column; skipping alignment plot.")

    # Prepare combined shared-axis plots (IRT, Baseline, optional Gemini)
    legend_handles, legend_labels = [], []
    shared_series = [
        plot_df['human_minutes'],
        plot_df['predicted_minutes'],
        plot_df['predicted_minutes_baseline']
    ]
    if not alignment_df.empty:
        shared_series.append(alignment_df['estimated_minutes'])

    shared_min, shared_max = compute_shared_limits(shared_series)

    fig_cols = 3 if not alignment_df.empty else 2
    fig, axes = plt.subplots(1, fig_cols, figsize=(7 * fig_cols, 6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    plot_prediction_vs_actual(
        axes[0],
        plot_df.dropna(subset=['predicted_minutes', 'human_minutes']),
        prediction_col='predicted_minutes',
        model_label='IRT',
        r_squared_value=r_squared_cybench,
        within_2x_pct=within_2x,
        shared_limits=(shared_min, shared_max),
        legend_handles=legend_handles,
        legend_labels=legend_labels
    )

    plot_prediction_vs_actual(
        axes[1],
        plot_df.dropna(subset=['predicted_minutes_baseline', 'human_minutes']),
        prediction_col='predicted_minutes_baseline',
        model_label='Baseline',
        r_squared_value=r_squared_baseline,
        within_2x_pct=within_2x_baseline,
        shared_limits=(shared_min, shared_max),
        legend_handles=legend_handles,
        legend_labels=legend_labels
    )

    if not alignment_df.empty:
        gemini_scatter = axes[2].scatter(
            alignment_df['human_minutes'],
            alignment_df['estimated_minutes'],
            c='#2c7fb8',
            alpha=0.8,
            s=60,
            label='Gemini estimated_minutes'
        )
        register_handle(gemini_scatter, gemini_scatter.get_label(), legend_handles, legend_labels)

        line_gemini, = axes[2].plot([shared_min, shared_max], [shared_min, shared_max],
                                    'k--', linewidth=2, label='Perfect Prediction')
        register_handle(line_gemini, line_gemini.get_label(), legend_handles, legend_labels)

        band_gemini = axes[2].fill_between([shared_min, shared_max], [shared_min/2, shared_max/2],
                                           [shared_min*2, shared_max*2], alpha=0.15, color='gray',
                                           label='2x Error Band')
        register_handle(band_gemini, band_gemini.get_label(), legend_handles, legend_labels)

        axes[2].set_title(
            f'Cybench: Actual FST vs Gemini Estimates\nR²={r_squared_gemini:.3f}, within 2x={within_2x_gemini:.1f}% (n={len(alignment_df)})',
            fontsize=13, fontweight='bold'
        )
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_xlim(shared_min, shared_max)
        axes[2].set_ylim(shared_min, shared_max)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_title(axes[1].get_title() + "\n(Gemini data unavailable)")

    fig.supxlabel('Actual FST (minutes)', fontsize=12)
    fig.supylabel('Predicted/Estimated Time (minutes)', fontsize=12)

    legend_ncol = max(3, int(np.ceil(len(legend_labels) / 2)))
    fig.legend(legend_handles, legend_labels, loc='lower center',
               fontsize=9, ncol=legend_ncol, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=(0, 0.08, 1, 1))

    output_path_combined_pdf = plots_dir / 'cybench_prediction_vs_actual_combined.pdf'
    output_path_combined_png = plots_dir / 'cybench_prediction_vs_actual_combined.png'
    plt.savefig(output_path_combined_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_combined_png, dpi=300, bbox_inches='tight')
    print(f"✓ Combined visualization saved to {output_path_combined_pdf}")
    print(f"✓ Combined visualization saved to {output_path_combined_png}")
    plt.show()

else:
    print("\nNo Cybench tasks with both difficulty estimates and FST annotations found.")
