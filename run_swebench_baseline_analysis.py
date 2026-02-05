#!/usr/bin/env python3
"""
Compare IRT vs Baseline predictions for SWE-bench verified tasks.
Computes misclassification rate against human time buckets.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare IRT vs Baseline predictions for SWE-bench')
parser.add_argument('--exclude-zero-success-data', action='store_true',
                    help='Exclude tasks with 0%% success rate at data collection stage')
parser.add_argument('--exclude-zero-success-plot', action='store_true',
                    help='Exclude tasks with 0%% success rate only at plotting stage')
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

parser.add_argument(
    "--params-path",
    type=Path,
    default=SCRIPT_DIR / "params" / "all_a_pyirt.csv",
    help="Path to the IRT parameters CSV (default: %(default)s)",
)
parser.add_argument(
    "--baseline-params-path",
    type=Path,
    default=SCRIPT_DIR / "params" / "all_a_pyirt_baseline.csv",
    help="Path to the baseline parameters CSV (default: %(default)s)",
)
args = parser.parse_args()

# Setup paths
BASE_DIR = SCRIPT_DIR
params_path = args.params_path
baseline_params_path = args.baseline_params_path
swe_pyirt_path = BASE_DIR / 'data' / 'swe_a_pyirt.jsonl'
plots_dir = BASE_DIR / 'plots'
plots_dir.mkdir(exist_ok=True)

# Load IRT parameters
print("Loading IRT parameters...")
df = pd.read_csv(params_path)
if df.columns[0] != 'task_id':
    df.rename(columns={df.columns[0]: 'task_id'}, inplace=True)

# Load baseline parameters
print("Loading baseline parameters...")
baseline_df = pd.read_csv(baseline_params_path)
if baseline_df.columns[0] != 'task_id':
    baseline_df.rename(columns={baseline_df.columns[0]: 'task_id'}, inplace=True)

# Merge baseline into main df
df = df.merge(baseline_df[['task_id', 'baseline_difficulty_logit', 'success_rate']],
              on='task_id', how='left')

# Exclude tasks with 0% success rate at data collection stage
if args.exclude_zero_success_data:
    zero_success_count = (df['success_rate'] == 0).sum()
    df = df[df['success_rate'] > 0]
    print(f"Excluded {zero_success_count} tasks with 0% success rate (data collection stage)")

# Load SWE-bench task IDs
print("Loading SWE-bench task IDs...")
swebench_task_ids = set()
with open(swe_pyirt_path) as f:
    for line in f:
        record = json.loads(line)
        swebench_task_ids.update(record['responses'].keys())

print(f"SWE-bench tasks: {len(swebench_task_ids)}")

# Mark task sources
df['task_source'] = 'other'
df.loc[df['task_id'].isin(swebench_task_ids), 'task_source'] = 'swebench'

# IRT prediction model (from METR-only fit)
slope_irt = 0.887408
intercept_irt = 2.877427

def predict_minutes_from_b(b_values):
    """Predict human minutes from IRT difficulty parameter b"""
    return np.exp(slope_irt * b_values + intercept_irt)

# Fit baseline prediction model using non-SWE-bench tasks with human_minutes
# Exclude tasks with 0 or negative human_minutes (would cause log(0) = -inf)
metr_mask = (df['task_source'] != 'swebench') & (df['human_minutes'].notna()) & (df['human_minutes'] > 0) & (df['baseline_difficulty_logit'].notna())
metr_data = df[metr_mask]

if len(metr_data) > 10:
    baseline_slope, baseline_intercept, r_value, p_value, std_err = stats.linregress(
        metr_data['baseline_difficulty_logit'],
        np.log(metr_data['human_minutes'])
    )
    print(f"\nBaseline model fit (on non-SWE data, n={len(metr_data)}):")
    print(f"  log(minutes) = {baseline_slope:.4f} * baseline_difficulty + {baseline_intercept:.4f}")
    print(f"  R² = {r_value**2:.3f}")
else:
    baseline_slope = slope_irt
    baseline_intercept = intercept_irt
    print("\nWarning: Not enough data for baseline fit, using IRT coefficients")

def predict_minutes_from_baseline(baseline_diff_values):
    """Predict human minutes from baseline difficulty (logit)"""
    return np.exp(baseline_slope * baseline_diff_values + baseline_intercept)

# Get SWE-bench predictions
swebench_predictions = df[(df['task_id'].isin(swebench_task_ids)) &
                          (df['task_source'] == 'swebench') &
                          (df['b'].notna())].copy()

print(f"\nSWE-bench tasks with IRT estimates: {len(swebench_predictions)}")

# Add predictions
swebench_predictions['predicted_minutes_irt'] = predict_minutes_from_b(swebench_predictions['b'])
swebench_predictions['predicted_minutes_baseline'] = predict_minutes_from_baseline(
    swebench_predictions['baseline_difficulty_logit']
)

# Filter to tasks with actual human_minutes
mask = swebench_predictions['human_minutes'].notna()
print(f"SWE-bench tasks with human time annotations: {mask.sum()}")

if mask.sum() == 0:
    print("No SWE-bench tasks with human time annotations found!")
    exit(1)

swebench_with_human = swebench_predictions[mask].copy()

# Define time buckets
bins = [0, 15, 60, 240, np.inf]
labels = ['<15 min', '15min-1hr', '1hr-4hrs', '>4hrs']

# Assign actual time buckets
swebench_with_human['actual_bucket'] = pd.cut(
    swebench_with_human['human_minutes'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Assign predicted time buckets (IRT)
swebench_with_human['predicted_bucket_irt'] = pd.cut(
    swebench_with_human['predicted_minutes_irt'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Assign predicted time buckets (Baseline)
swebench_with_human['predicted_bucket_baseline'] = pd.cut(
    swebench_with_human['predicted_minutes_baseline'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Compute misclassification
swebench_with_human['correct_irt'] = swebench_with_human['actual_bucket'] == swebench_with_human['predicted_bucket_irt']
swebench_with_human['correct_baseline'] = swebench_with_human['actual_bucket'] == swebench_with_human['predicted_bucket_baseline']

# Overall misclassification rate
misclass_irt = 1 - swebench_with_human['correct_irt'].mean()
misclass_baseline = 1 - swebench_with_human['correct_baseline'].mean()

print(f"\n{'='*60}")
print("MISCLASSIFICATION RATE (SWE-bench verified)")
print(f"{'='*60}")
print(f"{'Method':<20} {'Accuracy':>12} {'Misclassification':>18}")
print(f"{'-'*50}")
print(f"{'IRT':<20} {swebench_with_human['correct_irt'].mean()*100:>11.1f}% {misclass_irt*100:>17.1f}%")
print(f"{'Baseline':<20} {swebench_with_human['correct_baseline'].mean()*100:>11.1f}% {misclass_baseline*100:>17.1f}%")

# Per-bucket misclassification
print(f"\n{'='*60}")
print("PER-BUCKET ACCURACY")
print(f"{'='*60}")
print(f"{'Bucket':<15} {'Count':>8} {'IRT Acc':>12} {'Baseline Acc':>14}")
print(f"{'-'*50}")

for bucket in labels:
    bucket_mask = swebench_with_human['actual_bucket'] == bucket
    count = bucket_mask.sum()
    if count > 0:
        acc_irt = swebench_with_human.loc[bucket_mask, 'correct_irt'].mean() * 100
        acc_baseline = swebench_with_human.loc[bucket_mask, 'correct_baseline'].mean() * 100
        print(f"{bucket:<15} {count:>8} {acc_irt:>11.1f}% {acc_baseline:>13.1f}%")

# Confusion matrices
print(f"\n{'='*60}")
print("CONFUSION MATRIX - IRT")
print(f"{'='*60}")
conf_irt = pd.crosstab(swebench_with_human['actual_bucket'],
                        swebench_with_human['predicted_bucket_irt'],
                        margins=True)
print(conf_irt)

print(f"\n{'='*60}")
print("CONFUSION MATRIX - Baseline")
print(f"{'='*60}")
conf_baseline = pd.crosstab(swebench_with_human['actual_bucket'],
                             swebench_with_human['predicted_bucket_baseline'],
                             margins=True)
print(conf_baseline)

# Exclude tasks with 0% success rate at plotting stage (for analysis, keeps all data for stats)
swebench_with_human_plot = swebench_with_human.copy()
if args.exclude_zero_success_plot:
    zero_success_plot_count = (swebench_with_human_plot['success_rate'] == 0).sum()
    swebench_with_human_plot = swebench_with_human_plot[swebench_with_human_plot['success_rate'] > 0]
    print(f"\nExcluded {zero_success_plot_count} tasks with 0% success rate (plotting stage)")

# Compute plot-specific misclassification rates (may differ if --exclude-zero-success-plot is used)
if args.exclude_zero_success_plot:
    misclass_irt_plot = 1 - swebench_with_human_plot['correct_irt'].mean()
    misclass_baseline_plot = 1 - swebench_with_human_plot['correct_baseline'].mean()
    print(f"Plot data (after zero-success exclusion): {len(swebench_with_human_plot)} tasks")
    print(f"  IRT Accuracy: {(1-misclass_irt_plot)*100:.1f}%, Baseline Accuracy: {(1-misclass_baseline_plot)*100:.1f}%")
else:
    misclass_irt_plot = misclass_irt
    misclass_baseline_plot = misclass_baseline

# Create visualization with box plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Box plot styling
boxprops = dict(facecolor='#4c78a8', alpha=0.4, edgecolor='#26415f', linewidth=1.5)
whiskerprops = dict(color='#26415f', linewidth=1.2)
capprops = dict(color='#26415f', linewidth=1.2)
medianprops = dict(color='#26415f', linewidth=2)
flierprops = dict(marker='o', markerfacecolor='#4c78a8', markersize=4, alpha=0.5)

# Plot 1: IRT - Box plot of predicted times by actual time bucket
ax = axes[0]
sns.boxplot(
    data=swebench_with_human_plot,
    x='actual_bucket',
    y='predicted_minutes_irt',
    order=labels,
    ax=ax,
    width=0.5,
    boxprops=boxprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    medianprops=medianprops,
    flierprops=flierprops,
)

# Set log scale first to avoid issues with axhspan and 0
ax.set_yscale('log')

# Add bucket boundary lines
for b in bins[1:-1]:
    ax.axhline(y=b, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# Add shaded regions for each bucket (use small positive value instead of 0 for log scale)
y_min = swebench_with_human_plot['predicted_minutes_irt'].min() * 0.5
y_max = max(swebench_with_human_plot['predicted_minutes_irt'].max() * 2, 500)
ax.set_ylim(y_min, y_max)
ax.axhspan(y_min, 15, alpha=0.1, color='green', label='<15 min zone')
ax.axhspan(15, 60, alpha=0.1, color='blue', label='15min-1hr zone')
ax.axhspan(60, 240, alpha=0.1, color='orange', label='1hr-4hrs zone')
ax.axhspan(240, y_max, alpha=0.1, color='red', label='>4hrs zone')

ax.set_xlabel('Actual Time Bucket', fontsize=11)
ax.set_ylabel('IRT Predicted Time (minutes)', fontsize=11)
ax.set_title(f'IRT: Predicted Time by Actual Bucket\nBucket Accuracy: {(1-misclass_irt_plot)*100:.1f}%', fontsize=12, fontweight='bold')
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add sample size annotations
for i, bucket in enumerate(labels):
    n = (swebench_with_human_plot['actual_bucket'] == bucket).sum()
    if n > 0:
        acc = swebench_with_human_plot.loc[swebench_with_human_plot['actual_bucket'] == bucket, 'correct_irt'].mean() * 100
        ax.text(i, ax.get_ylim()[0] * 1.5, f'n={n}\n{acc:.0f}%', ha='center', fontsize=9, va='bottom')
    else:
        ax.text(i, ax.get_ylim()[0] * 1.5, f'n=0', ha='center', fontsize=9, va='bottom')

# Plot 2: Baseline - Box plot of predicted times by actual time bucket
ax = axes[1]

# Different color for baseline
boxprops_baseline = dict(facecolor='#e45756', alpha=0.4, edgecolor='#8b0000', linewidth=1.5)
whiskerprops_baseline = dict(color='#8b0000', linewidth=1.2)
capprops_baseline = dict(color='#8b0000', linewidth=1.2)
medianprops_baseline = dict(color='#8b0000', linewidth=2)
flierprops_baseline = dict(marker='o', markerfacecolor='#e45756', markersize=4, alpha=0.5)

# Filter to only data with valid baseline predictions
baseline_plot_data = swebench_with_human_plot[swebench_with_human_plot['predicted_minutes_baseline'].notna()]
# Only use labels for which we have data
baseline_labels = [l for l in labels if (baseline_plot_data['actual_bucket'] == l).any()]

if len(baseline_plot_data) > 0 and len(baseline_labels) > 0:
    sns.boxplot(
        data=baseline_plot_data,
        x='actual_bucket',
        y='predicted_minutes_baseline',
        order=baseline_labels,
        ax=ax,
        width=0.5,
        boxprops=boxprops_baseline,
        whiskerprops=whiskerprops_baseline,
        capprops=capprops_baseline,
        medianprops=medianprops_baseline,
        flierprops=flierprops_baseline,
    )
else:
    ax.text(0.5, 0.5, 'No baseline data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)

# Only set up plot details if we have baseline data
if len(baseline_plot_data) > 0:
    # Set log scale first to avoid issues with axhspan and 0
    ax.set_yscale('log')

    # Add bucket boundary lines
    for b in bins[1:-1]:
        ax.axhline(y=b, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Add shaded regions for each bucket (use small positive value instead of 0 for log scale)
    y_min_baseline = baseline_plot_data['predicted_minutes_baseline'].min() * 0.5
    y_max_baseline = max(baseline_plot_data['predicted_minutes_baseline'].max() * 2, 500)
    ax.set_ylim(y_min_baseline, y_max_baseline)
    ax.axhspan(y_min_baseline, 15, alpha=0.1, color='green')
    ax.axhspan(15, 60, alpha=0.1, color='blue')
    ax.axhspan(60, 240, alpha=0.1, color='orange')
    ax.axhspan(240, y_max_baseline, alpha=0.1, color='red')

    ax.set_xlabel('Actual Time Bucket', fontsize=11)
    ax.set_ylabel('Baseline Predicted Time (minutes)', fontsize=11)
    ax.set_title(f'Baseline: Predicted Time by Actual Bucket\nBucket Accuracy: {(1-misclass_baseline_plot)*100:.1f}%', fontsize=12, fontweight='bold')
    ax.set_xticklabels(baseline_labels, rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add sample size annotations
    for i, bucket in enumerate(baseline_labels):
        n = (baseline_plot_data['actual_bucket'] == bucket).sum()
        if n > 0:
            acc = baseline_plot_data.loc[baseline_plot_data['actual_bucket'] == bucket, 'correct_baseline'].mean() * 100
            ax.text(i, ax.get_ylim()[0] * 1.5, f'n={n}\n{acc:.0f}%', ha='center', fontsize=9, va='bottom')
        else:
            ax.text(i, ax.get_ylim()[0] * 1.5, f'n=0', ha='center', fontsize=9, va='bottom')
else:
    ax.set_xlabel('Actual Time Bucket', fontsize=11)
    ax.set_ylabel('Baseline Predicted Time (minutes)', fontsize=11)
    ax.set_title('Baseline: No Data Available', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save plot
output_path = plots_dir / 'swebench_irt_vs_baseline_misclassification.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to {output_path}")

output_path_pdf = plots_dir / 'swebench_irt_vs_baseline_misclassification.pdf'
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to {output_path_pdf}")

plt.close()

# Additional metrics
print(f"\n{'='*60}")
print("ADDITIONAL METRICS")
print(f"{'='*60}")

# Filter to rows with valid predictions for metrics
valid_irt = swebench_with_human['predicted_minutes_irt'].notna()
valid_baseline = swebench_with_human['predicted_minutes_baseline'].notna()
valid_both = valid_irt & valid_baseline

# R² on log scale (IRT)
y_actual_irt = np.log(swebench_with_human.loc[valid_irt, 'human_minutes'])
y_pred_irt = np.log(swebench_with_human.loc[valid_irt, 'predicted_minutes_irt'])
ss_tot_irt = np.sum((y_actual_irt - y_actual_irt.mean()) ** 2)
r2_irt = 1 - np.sum((y_actual_irt - y_pred_irt) ** 2) / ss_tot_irt if ss_tot_irt > 0 else np.nan

# R² on log scale (Baseline)
if valid_baseline.sum() > 0:
    y_actual_baseline = np.log(swebench_with_human.loc[valid_baseline, 'human_minutes'])
    y_pred_baseline = np.log(swebench_with_human.loc[valid_baseline, 'predicted_minutes_baseline'])
    ss_tot_baseline = np.sum((y_actual_baseline - y_actual_baseline.mean()) ** 2)
    r2_baseline = 1 - np.sum((y_actual_baseline - y_pred_baseline) ** 2) / ss_tot_baseline if ss_tot_baseline > 0 else np.nan
else:
    r2_baseline = np.nan

# MAE on log scale
mae_irt = np.mean(np.abs(y_actual_irt - y_pred_irt)) if valid_irt.sum() > 0 else np.nan
mae_baseline = np.mean(np.abs(y_actual_baseline - y_pred_baseline)) if valid_baseline.sum() > 0 else np.nan

# Within 2x
ratio_irt = swebench_with_human.loc[valid_irt, 'predicted_minutes_irt'] / swebench_with_human.loc[valid_irt, 'human_minutes']
within_2x_irt = ((ratio_irt >= 0.5) & (ratio_irt <= 2.0)).mean() * 100 if valid_irt.sum() > 0 else np.nan

if valid_baseline.sum() > 0:
    ratio_baseline = swebench_with_human.loc[valid_baseline, 'predicted_minutes_baseline'] / swebench_with_human.loc[valid_baseline, 'human_minutes']
    within_2x_baseline = ((ratio_baseline >= 0.5) & (ratio_baseline <= 2.0)).mean() * 100
else:
    within_2x_baseline = np.nan

print(f"{'Metric':<25} {'IRT':>12} {'Baseline':>12}")
print(f"{'-'*50}")
print(f"{'R² (log scale)':<25} {r2_irt:>12.3f} {r2_baseline:>12.3f}")
print(f"{'MAE (log scale)':<25} {mae_irt:>12.3f} {mae_baseline:>12.3f}")
print(f"{'Within 2x (%)':<25} {within_2x_irt:>12.1f} {within_2x_baseline:>12.1f}")
print(f"{'Bucket Accuracy (%)':<25} {(1-misclass_irt)*100:>12.1f} {(1-misclass_baseline)*100:>12.1f}")
