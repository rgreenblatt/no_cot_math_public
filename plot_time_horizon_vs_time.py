#!/usr/bin/env python3
"""
Plot single forward pass math time horizon vs model release date.
Shows exponential growth in model capabilities over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from scipy.optimize import curve_fit

# Model data: (name, release_date, time_horizon_minutes, include_in_fit)
MODELS = [
    ("GPT-4", datetime(2023, 3, 14), 0.4, True),
    ("GPT-4o", datetime(2024, 5, 13), 0.5, False),
    ("Opus 3", datetime(2024, 3, 4), 0.7, True),
    ("Sonnet 3.5", datetime(2024, 6, 20), 0.9, True),
    ("Sonnet 3.6", datetime(2024, 10, 22), 0.8, False),
    ("Sonnet 3.7", datetime(2025, 2, 24), 0.9, False),
    ("Sonnet 4", datetime(2025, 5, 22), 1.6, False),
    ("Sonnet 4.5", datetime(2025, 9, 29), 2.0, False),
    ("GPT-4.1", datetime(2025, 4, 14), 0.7, False),
    ("Opus 4", datetime(2025, 5, 22), 2.3, True),
    ("Opus 4.5", datetime(2025, 11, 24), 3.5, True),
    ("GPT-5.1", datetime(2025, 11, 12), 0.6, False),
    ("GPT-5.2", datetime(2025, 12, 11), 1.2, False),
    ("deepseek-v3", datetime(2024, 12, 27), 1.1, False),
    # ("deepseek-v3.2", datetime(2025, 12, 1), 1.2, False),
    ("qwen3-235b-a22b", datetime(2025, 7, 21), 1.5, False),
    # ("kimi-k2", datetime(2025, 7, 15), 1.3, False),
]

MODELS_NO_REPEAT = [
    # ("GPT-4", datetime(2023, 3, 14), 0.4, True),
    ("Opus 3", datetime(2024, 3, 4), 0.5, True),
    ("Sonnet 3.5", datetime(2024, 6, 20), 0.6, True),
    ("Opus 4", datetime(2025, 5, 22), 1.7, True),
    ("Opus 4.5", datetime(2025, 11, 24), 2.6, True),
]

# Reference date for calculating days
REFERENCE_DATE = datetime(2023, 1, 1)


def days_since_reference(date):
    """Convert date to days since reference date."""
    return (date - REFERENCE_DATE).days


def exponential(x, a, b):
    """Exponential function: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def main():
    # Extract data
    names = [m[0] for m in MODELS]
    dates = [m[1] for m in MODELS]
    time_horizons = np.array([m[2] for m in MODELS])
    include_in_fit = [m[3] for m in MODELS]

    # Convert dates to numeric (days since reference)
    days = np.array([days_since_reference(d) for d in dates])

    # Filter data for fitting (only models with include_in_fit=True)
    fit_mask = np.array(include_in_fit)
    days_fit = days[fit_mask]
    time_horizons_fit = time_horizons[fit_mask]

    # Fit exponential curve (only on included models)
    popt, pcov = curve_fit(exponential, days_fit, time_horizons_fit, p0=[0.1, 0.001])
    a, b = popt

    # Calculate doubling time (days for time horizon to double)
    doubling_time_days = np.log(2) / b
    doubling_time_months = doubling_time_days / 30.44  # average days per month

    print(f"Exponential fit: y = {a:.4f} * exp({b:.6f} * x)")
    print(f"Doubling time: {doubling_time_days:.0f} days ({doubling_time_months:.1f} months)")

    # Generate smooth curve for fit
    days_range = np.linspace(days.min() - 30, days.max() + 60, 200)
    dates_range = [REFERENCE_DATE + np.timedelta64(int(d), 'D') for d in days_range]
    fit_values = exponential(days_range, a, b)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Separate data for plotting
    dates_fit = [d for d, inc in zip(dates, include_in_fit) if inc]
    dates_nofit = [d for d, inc in zip(dates, include_in_fit) if not inc]
    th_fit = [th for th, inc in zip(time_horizons, include_in_fit) if inc]
    th_nofit = [th for th, inc in zip(time_horizons, include_in_fit) if not inc]
    names_fit = [n for n, inc in zip(names, include_in_fit) if inc]
    names_nofit = [n for n, inc in zip(names, include_in_fit) if not inc]

    # --- Plot 1: Linear y-axis with exponential fit ---
    ax1.scatter(dates_fit, th_fit, s=100, c='blue', zorder=5, label='Models (frontier)')
    ax1.scatter(dates_nofit, th_nofit, s=100, c='lightblue', zorder=5, label='Models (non-frontier)')
    ax1.plot(dates_range, fit_values, 'r--', linewidth=2, alpha=0.7,
             label=f'Exponential fit (doubling: {doubling_time_months:.1f} months)')

    # Add labels for each point
    for name, date, th, inc in zip(names, dates, time_horizons, include_in_fit):
        color = 'black' if inc else 'gray'
        # Place qwen label below the dot to avoid intersection
        y_offset = -15 if name.startswith('qwen') else 10
        ax1.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, y_offset), ha='center', fontsize=10, fontweight='bold', color=color)

    ax1.set_xlabel('Release Date', fontsize=12)
    ax1.set_ylabel('Time Horizon (minutes)', fontsize=12)
    ax1.set_title('No Chain-of-Thought Math Time Horizon vs Time\n(Linear Scale)', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(time_horizons) * 1.3)
    ax1.set_yticks([0.5, 1, 2, 3, 4])

    # Format x-axis dates
    ax1.tick_params(axis='x', rotation=30)

    # --- Plot 2: Log y-axis (exponential becomes linear) ---
    ax2.scatter(dates_fit, th_fit, s=100, c='blue', zorder=5, label='Models (frontier)')
    ax2.scatter(dates_nofit, th_nofit, s=100, c='lightblue', zorder=5, label='Models (non-frontier)')
    ax2.plot(dates_range, fit_values, 'r--', linewidth=2, alpha=0.7,
             label=f'Exponential fit (doubling: {doubling_time_months:.1f} months)')

    # Add labels for each point
    for name, date, th, inc in zip(names, dates, time_horizons, include_in_fit):
        color = 'black' if inc else 'gray'
        # Place qwen label below the dot to avoid intersection
        y_offset = -15 if name.startswith('qwen') else 10
        ax2.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, y_offset), ha='center', fontsize=10, fontweight='bold', color=color)

    ax2.set_xlabel('Release Date', fontsize=12)
    ax2.set_ylabel('Time Horizon (minutes, log scale)', fontsize=12)
    ax2.set_title('No Chain-of-Thought Math Time Horizon vs Time\n(Log Scale - Exponential appears linear)', fontsize=13)
    ax2.set_yscale('log', base=2)
    ax2.set_yticks([0.5, 1, 2, 3, 4])
    ax2.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, which='major')

    # Format x-axis dates
    ax2.tick_params(axis='x', rotation=30)

    # Set y-axis limits for log scale
    ax2.set_ylim(0.25, 5)

    plt.tight_layout()

    # Save plot
    output_path = 'eval_results/time_horizon_vs_release_date.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # --- New comparison plot: MODELS vs MODELS_NO_REPEAT ---
    # Extract MODELS_NO_REPEAT data (only include_in_fit=True)
    names_nr = [m[0] for m in MODELS_NO_REPEAT if m[3]]
    dates_nr = [m[1] for m in MODELS_NO_REPEAT if m[3]]
    time_horizons_nr = np.array([m[2] for m in MODELS_NO_REPEAT if m[3]])
    days_nr = np.array([days_since_reference(d) for d in dates_nr])

    # Fit exponential curve for no repeat data
    popt_nr, _ = curve_fit(exponential, days_nr, time_horizons_nr, p0=[0.1, 0.001])
    a_nr, b_nr = popt_nr
    doubling_time_days_nr = np.log(2) / b_nr
    doubling_time_months_nr = doubling_time_days_nr / 30.44

    print(f"\nNo repeat fit: y = {a_nr:.4f} * exp({b_nr:.6f} * x)")
    print(f"No repeat doubling time: {doubling_time_days_nr:.0f} days ({doubling_time_months_nr:.1f} months)")

    # Generate fit curve for no repeat
    fit_values_nr = exponential(days_range, a_nr, b_nr)

    # Create comparison figure
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7))

    # --- Comparison Plot 1: Linear scale ---
    # Plot MODELS (frontier only)
    ax3.scatter(dates_fit, th_fit, s=100, c='blue', zorder=5, label='With repeat/filler')
    ax3.plot(dates_range, fit_values, 'b--', linewidth=2, alpha=0.7,
             label=f'Fit (doubling: {doubling_time_months:.1f} months)')

    # Plot MODELS_NO_REPEAT
    ax3.scatter(dates_nr, time_horizons_nr, s=100, c='orange', zorder=5, label='No repeat or filler')
    ax3.plot(dates_range, fit_values_nr, color='orange', linestyle='--', linewidth=2, alpha=0.7,
             label=f'Fit (doubling: {doubling_time_months_nr:.1f} months)')

    # Add labels for MODELS (frontier only)
    for name, date, th in zip(names_fit, dates_fit, th_fit):
        ax3.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='blue')

    # Add labels for MODELS_NO_REPEAT
    for name, date, th in zip(names_nr, dates_nr, time_horizons_nr):
        ax3.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=9, fontweight='bold', color='orange')

    ax3.set_xlabel('Release Date', fontsize=12)
    ax3.set_ylabel('Time Horizon (minutes)', fontsize=12)
    ax3.set_title('Comparison: With vs Without Repeat/Filler\n(Linear Scale)', fontsize=13)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(max(th_fit), max(time_horizons_nr)) * 1.3)
    ax3.set_yticks([0.5, 1, 2, 3, 4])
    ax3.tick_params(axis='x', rotation=30)

    # --- Comparison Plot 2: Log scale ---
    ax4.scatter(dates_fit, th_fit, s=100, c='blue', zorder=5, label='With repeat/filler')
    ax4.plot(dates_range, fit_values, 'b--', linewidth=2, alpha=0.7,
             label=f'Fit (doubling: {doubling_time_months:.1f} months)')

    ax4.scatter(dates_nr, time_horizons_nr, s=100, c='orange', zorder=5, label='No repeat or filler')
    ax4.plot(dates_range, fit_values_nr, color='orange', linestyle='--', linewidth=2, alpha=0.7,
             label=f'Fit (doubling: {doubling_time_months_nr:.1f} months)')

    # Add labels
    for name, date, th in zip(names_fit, dates_fit, th_fit):
        ax4.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='blue')

    for name, date, th in zip(names_nr, dates_nr, time_horizons_nr):
        ax4.annotate(name, (date, th), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=9, fontweight='bold', color='orange')

    ax4.set_xlabel('Release Date', fontsize=12)
    ax4.set_ylabel('Time Horizon (minutes, log scale)', fontsize=12)
    ax4.set_title('Comparison: With vs Without Repeat/Filler\n(Log Scale)', fontsize=13)
    ax4.set_yscale('log', base=2)
    ax4.set_yticks([0.5, 1, 2, 3, 4])
    ax4.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, which='major')
    ax4.set_ylim(0.25, 5)
    ax4.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    # Save comparison plot
    output_path2 = 'eval_results/time_horizon_comparison.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path2}")

    # Also display
    plt.show()

    # Print summary table
    print("\n" + "="*60)
    print("MODEL TIME HORIZONS (Single Forward Pass)")
    print("="*60)
    print(f"{'Model':<15} {'Release Date':<15} {'Time Horizon':<15}")
    print("-"*45)
    for name, date, th in zip(names, dates, time_horizons):
        print(f"{name:<15} {date.strftime('%Y-%m-%d'):<15} {th:.1f} min")
    print("-"*45)
    print(f"\nExponential fit parameters:")
    print(f"  y = {a:.4f} * exp({b:.6f} * days)")
    print(f"  Doubling time: {doubling_time_months:.1f} months")


if __name__ == "__main__":
    main()
