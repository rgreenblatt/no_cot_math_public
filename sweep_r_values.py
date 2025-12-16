#!/usr/bin/env python3
"""
Sweep through different r values (problem repetitions) for multiple models
and plot the performance.
"""

import asyncio
import json
import os
from eval_math_no_reasoning import run_evaluation, parse_model_name
from analyze_results import fit_time_horizon, load_difficulty_ratings
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


async def sweep_evaluations(
    models: list[str],
    r_values: list[int | None],
    sweep_for_arith: bool = True,
    verbosity: int = 2,
    max_problems: int | None = None,
    load_from_output: bool = False,
) -> dict:
    """Run evaluations for all combinations of models and r values."""

    # Configuration
    if sweep_for_arith:
        input_file = "arithmetic_problems.jsonl"
        output_prefix = "arith_sweep"
    else:
        input_file = "problems_with_answers.jsonl"
        output_prefix = "sweep"

    concurrency = 5

    # Create output directory if it doesn't exist
    os.makedirs("eval_results", exist_ok=True)

    # Store results for plotting
    all_results = {}

    for model in models:
        model_results = {}
        model_id = parse_model_name(model)
        k_shot = 3 if model == "gpt-4" else 10

        for r in r_values:
            r_label = 1 if r is None else r
            output_file = f"eval_results/{output_prefix}_{model}_r{r_label}.json"

            if verbosity >= 2:
                print(f"\n{'='*80}")
                print(f"Evaluating {model} with r={r_label} on {input_file}")
                print(f"{'='*80}")

            # Run evaluation
            results = await run_evaluation(
                input_file=input_file,
                output_file=output_file,
                max_problems=max_problems,
                concurrency=(concurrency // 3 if r is not None and r >= 20 else concurrency),
                model=model_id,
                repeat_problem=r,
                verbosity=verbosity,
                load_from_output=load_from_output,
                k_shot=k_shot,
            )

            # Calculate accuracy
            correct_count = sum(1 for result in results if result["is_correct"])
            accuracy = correct_count / len(results) if results else 0

            # Store accuracy and sample size
            model_results[r_label] = {
                "accuracy": accuracy,
                "n": len(results),
                "correct": correct_count,
            }

            if verbosity >= 2:
                print(f"Accuracy for {model} with r={r_label}: {accuracy:.2%}")

        all_results[model] = model_results

    # Save combined results
    with open("eval_results/sweep_combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


async def sweep_filler_evaluations(
    models: list[str],
    f_values: list[int | None],
    sweep_for_arith: bool = True,
    verbosity: int = 2,
    max_problems: int | None = None,
    load_from_output: bool = False,
) -> dict:
    """Run evaluations for all combinations of models and filler token values."""

    # Configuration
    if sweep_for_arith:
        input_file = "arithmetic_problems.jsonl"
        output_prefix = "arith_sweep_filler"
    else:
        input_file = "problems_with_answers.jsonl"
        output_prefix = "sweep_filler"

    concurrency = 5

    # Create output directory if it doesn't exist
    os.makedirs("eval_results", exist_ok=True)

    # Store results for plotting
    all_results = {}

    for model in models:
        model_results = {}
        model_id = parse_model_name(model)
        k_shot = 3 if model == "gpt-4" else 10

        for f in f_values:
            f_label = 0 if f is None else f
            output_file = f"eval_results/{output_prefix}_{model}_f{f_label}.json"

            if verbosity >= 2:
                print(f"\n{'='*80}")
                print(f"Evaluating {model} with filler_tokens={f_label} on {input_file}")
                print(f"{'='*80}")

            # Run evaluation
            results = await run_evaluation(
                input_file=input_file,
                output_file=output_file,
                max_problems=max_problems,
                concurrency=(concurrency // 3 if f is not None and f >= 500 else concurrency),
                model=model_id,
                repeat_problem=None,
                filler_tokens=f,
                verbosity=verbosity,
                load_from_output=load_from_output,
                k_shot=k_shot,
            )

            # Calculate accuracy
            correct_count = sum(1 for result in results if result["is_correct"])
            accuracy = correct_count / len(results) if results else 0

            # Store accuracy and sample size
            model_results[f_label] = {
                "accuracy": accuracy,
                "n": len(results),
                "correct": correct_count,
            }

            if verbosity >= 2:
                print(f"Accuracy for {model} with filler_tokens={f_label}: {accuracy:.2%}")

        all_results[model] = model_results

    # Save combined results
    with open("eval_results/sweep_filler_combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def generate_markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Generate a markdown table from headers and rows."""
    lines = [f"## {title}", ""]

    # Header row
    lines.append("| " + " | ".join(headers) + " |")
    # Separator row
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # Data rows
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def plot_results(results, r_values_disp, sweep_for_arith: bool = True, sweep_type: str = "r"):
    """Create a plot showing performance by model and r/filler value with error bars."""

    # Setup plot
    plt.figure(figsize=(12, 8))

    # Colors for different models
    colors = {
        "opus-4-5": "#FF6B6B",
        "opus-4": "#F38181",
        "sonnet-4-5": "#95E1D3",
        "sonnet-4": "#4ECDC4",
        "haiku-3-5": "#AA96DA",
        "haiku-3": "#F7D794",
    }

    # Markers for different models
    markers = {
        "opus-4-5": "o",
        "sonnet-4-5": "s",
        "opus-4": "^",
        "sonnet-4": "D",
        "haiku-3-5": "v",
        "haiku-3": "P",
    }

    # Plot each model
    for model, model_results in results.items():
        r_values = sorted(model_results.keys())

        # Extract accuracies and calculate error bars
        accuracies = []
        error_bars = []

        for r in r_values:
            # Handle both old format (just accuracy) and new format (dict with accuracy, n, correct)
            if isinstance(model_results[r], dict):
                accuracy = model_results[r]["accuracy"]
                n = model_results[r]["n"]
            else:
                # Old format - assume 600 problems
                accuracy = model_results[r]
                n = 600

            accuracies.append(accuracy * 100)  # Convert to percentage

            # Calculate 95% confidence interval using normal approximation
            # CI = 1.96 * sqrt(p * (1-p) / n)
            if accuracy == 0 or accuracy == 1:
                # Use Wilson score interval for edge cases
                error = 1.96 * np.sqrt((accuracy * (1 - accuracy) + 1 / (4 * n)) / n) * 100
            else:
                error = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n) * 100

            error_bars.append(error)

        # Use indices for x positions to get even spacing
        x_values = list(range(len(r_values)))

        # Plot with error bars
        plt.errorbar(
            x_values,
            accuracies,
            yerr=error_bars,
            marker=markers[model],
            color=colors[model],
            label=model,
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=1.5,
            alpha=0.9,
        )

    # Customize plot
    if sweep_type == "r":
        plt.xlabel("Problem Repetitions (r)", fontsize=12, fontweight="bold")
        title_var = "Problem Repetitions"
    else:
        plt.xlabel("Filler Tokens (f)", fontsize=12, fontweight="bold")
        title_var = "Filler Tokens"
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    dataset_name = "Gen-Arithmetic" if sweep_for_arith else "Easy-Comp-Math"
    plt.title(
        f"Model Performance vs {title_var} (with 95% CI)\n{dataset_name} Dataset",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3, linestyle="--")

    # Set x-axis to show all values with even spacing
    x_positions = list(range(len(r_values_disp)))
    plt.xticks(x_positions, [str(v) for v in r_values_disp])

    # Add some padding to y-axis
    plt.ylim(bottom=0, top=100)

    # Save plot
    plt.tight_layout()
    prefix = "arith_" if sweep_for_arith else ""
    filler_suffix = "_filler" if sweep_type == "f" else ""
    filename = f"eval_results/{prefix}sweep{filler_suffix}_performance_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {filename}")

    # plt.show()


def two_proportion_ztest(x1, n1, x2, n2):
    """
    Perform two-proportion z-test.

    Args:
        x1: number of successes in sample 1
        n1: size of sample 1
        x2: number of successes in sample 2
        n2: size of sample 2

    Returns:
        p-value for two-tailed test
    """
    p1 = x1 / n1
    p2 = x2 / n2

    # Pooled proportion
    p_pool = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    # Avoid division by zero
    if se == 0:
        return 1.0

    # Z-statistic
    z = (p1 - p2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_value

def load_problem_data(model, value_label, sweep_for_arith: bool = True, sweep_type: str = "r"):
    """Load individual problem results from output file."""
    if sweep_for_arith:
        output_prefix = "arith_sweep_filler" if sweep_type == "f" else "arith_sweep"
    else:
        output_prefix = "sweep_filler" if sweep_type == "f" else "sweep"

    param_prefix = "f" if sweep_type == "f" else "r"
    output_file = f"eval_results/{output_prefix}_{model}_{param_prefix}{value_label}.json"

    with open(output_file, "r") as f:
        data = json.load(f)

    return data

def load_problem_results(model, value_label, sweep_for_arith: bool = True, sweep_type: str = "r"):
    data = load_problem_data(model, value_label, sweep_for_arith, sweep_type)

    # Return list of per-problem correctness (1 if correct, 0 if incorrect)
    # Indexed by problem_index
    results = {}
    problems = {}
    for result in data["results"]:
        problem_idx = result["problem_index"]
        results[problem_idx] = 1 if result["is_correct"] else 0
        problems[problem_idx] = result["problem"]

    return results, problems


def paired_ttest_on_problems(model, v1_label, v2_label, sweep_for_arith: bool = True, sweep_type: str = "r"):
    """
    Perform paired t-test on problem-level results between two r/filler values.

    Returns:
        (mean_diff, p_value) where mean_diff is v1 - v2 in percentage points
    """
    # Load results for both values
    results_v1, problems_v1 = load_problem_results(model, v1_label, sweep_for_arith, sweep_type)
    results_v2, problems_v2 = load_problem_results(model, v2_label, sweep_for_arith, sweep_type)

    assert problems_v1 == problems_v2  # Ensure same problems

    assert set(results_v1.keys()) == set(results_v2.keys())  # for now, assert run on the same problems

    # Get common problem indices
    common_indices = sorted(results_v1.keys())

    if len(common_indices) == 0:
        return 0.0, 1.0

    # Create paired arrays
    scores_v1 = np.array([results_v1[idx] for idx in common_indices])
    scores_v2 = np.array([results_v2[idx] for idx in common_indices])

    # Calculate mean difference in percentage points
    mean_diff = (scores_v1.mean() - scores_v2.mean()) * 100

    # Perform paired t-test
    if np.array_equal(scores_v1, scores_v2):
        # Identical results
        p_value = 1.0
    else:
        t_stat, p_value = stats.ttest_rel(scores_v1, scores_v2)

    return mean_diff, p_value


def plot_significance_matrices(results, values_disp, sweep_for_arith: bool = True, sweep_type: str = "r"):
    """
    Create significance matrices for each model showing pairwise comparisons
    between different r/filler values (lower triangle only).
    """

    param_label = "r" if sweep_type == "r" else "f"

    for model, model_results in results.items():
        n_vals = len(values_disp)

        # Create matrices for p-values and accuracy differences
        p_matrix = np.ones((n_vals, n_vals))
        diff_matrix = np.zeros((n_vals, n_vals))

        # Calculate pairwise comparisons using paired t-test
        for i, v1 in enumerate(values_disp):
            for j, v2 in enumerate(values_disp):
                if i <= j:  # Skip upper triangle and diagonal
                    continue

                # Perform paired t-test on problem-level results
                mean_diff, p_value = paired_ttest_on_problems(model, v1, v2, sweep_for_arith, sweep_type)

                diff_matrix[i, j] = mean_diff
                p_matrix[i, j] = p_value

        # Create single figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Mask upper triangle and diagonal
        mask = np.triu(np.ones((n_vals, n_vals), dtype=bool))

        # Create annotations showing p-values (only for lower triangle)
        annot_p = np.empty((n_vals, n_vals), dtype=object)
        for i in range(n_vals):
            for j in range(n_vals):
                if i <= j:
                    annot_p[i, j] = ""
                else:
                    p = p_matrix[i, j]
                    if p < 0.01:
                        annot_p[i, j] = f"{p:.2e}"
                    else:
                        annot_p[i, j] = f"{p:.3f}"

        # Use diverging colormap centered at 0 for accuracy difference
        # Only consider lower triangle for color scale
        lower_triangle_values = diff_matrix[~mask]
        if len(lower_triangle_values) > 0:
            max_abs_diff = np.max(np.abs(lower_triangle_values))
        else:
            max_abs_diff = 1.0

        sns.heatmap(
            diff_matrix,
            annot=annot_p,
            fmt="s",
            cmap="RdBu_r",
            center=0,
            vmin=-max_abs_diff,
            vmax=max_abs_diff,
            xticklabels=[f"{param_label}={v}" for v in values_disp],
            yticklabels=[f"{param_label}={v}" for v in values_disp],
            cbar_kws={"label": "Accuracy Difference (percentage points, row - column)"},
            mask=mask,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(
            f"{model}: Accuracy Difference (color) and p-value (text)\nrow - column, paired t-test",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlabel(f"{param_label} value (column)", fontweight="bold")
        ax.set_ylabel(f"{param_label} value (row)", fontweight="bold")

        plt.tight_layout()

        # Save plot
        prefix = "arith_" if sweep_for_arith else ""
        filler_suffix = "_filler" if sweep_type == "f" else ""
        filename = f"eval_results/{prefix}significance_matrix{filler_suffix}_{model}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Significance matrix saved to: {filename}")

        plt.close()


async def run_sweep(sweep_type: str, sweep_for_arith: bool, verbosity: int = 0):
    if sweep_type == "f" and sweep_for_arith:
        max_problems = 600
    else:
        max_problems = None

    models = ["opus-4-5", "opus-4", "sonnet-4-5", "sonnet-4", "haiku-3-5", "haiku-3"]
    # models = ["opus-4-5", "opus-4"]

    if sweep_for_arith:
        if sweep_type == "r":
            values = [None, 2, 3, 5, 10, 20, 40]
        else:
            values = [None, 30, 100, 300, 1000]
    else:
        if sweep_type == "r":
            values = [None, 2, 3, 5, 10]
        else:
            values = [None, 30, 100, 300, 1000, 3000]

    # Run evaluations
    if sweep_type == "r":
        results = await sweep_evaluations(
            models, values, sweep_for_arith, verbosity=verbosity, max_problems=max_problems, load_from_output=True,
        )
        values_disp = [1 if v is None else v for v in values]
        param_label = "r"
    else:
        results = await sweep_filler_evaluations(
            models, values, sweep_for_arith, verbosity=verbosity, max_problems=max_problems, load_from_output=True,
        )
        values_disp = [0 if v is None else v for v in values]
        param_label = "f"

    # Print summary table
    dataset_name = "Gen-Arithmetic" if sweep_for_arith else "Easy-Comp-Math"
    sweep_type_name = "Repetitions" if sweep_type == "r" else "Filler"
    print("\n" + "=" * 80)
    print(f"SUMMARY TABLE - {dataset_name} - {sweep_type_name}")
    print("=" * 80)
    print(f"{'Model':<15} | " + " | ".join([f"{param_label}={v:>3}" for v in values_disp]))
    print("-" * 80)

    # Build markdown table
    headers = ["Model"] + [f"{param_label}={v}" for v in values_disp]
    rows = []

    for model in models:
        if model in results:
            accuracies = []
            for v in values_disp:
                result = results[model].get(v, 0)
                if isinstance(result, dict):
                    acc = result["accuracy"] * 100
                else:
                    acc = result * 100
                accuracies.append(f"{acc:.1f}%")
            print(f"{model:<15} | " + " | ".join([f"{a:>6}" for a in accuracies]))
            rows.append([model] + accuracies)

    # Create plot
    plot_results(results, values_disp, sweep_for_arith, sweep_type)

    # Create significance matrices
    plot_significance_matrices(results, values_disp, sweep_for_arith, sweep_type)

    # Return markdown table
    return generate_markdown_table(f"{dataset_name} - {sweep_type_name}", headers, rows)


async def run_partial_sweep(sweep_type: str, sweep_for_arith: bool, verbosity: int = 0):
    """Run a partial sweep for non-Anthropic models with limited r/f values."""
    if sweep_for_arith:
        max_problems = 600
    else:
        max_problems = None

    partial_sweep_models = [
        "opus-4-5",
        "opus-4",
        "sonnet-4-5",
        "sonnet-4",
        "haiku-3-5",
        "haiku-3",
        "haiku-4-5",
        "gpt-3.5",
        "gpt-4", # note: GPT-4 uses k_shot=3
        "gpt-4o",
        "gpt-4.1",
        "gpt-5.1",
        "gpt-5.2",
        "deepseek-v3",
        "qwen3-235b-a22b",
    ]

    if sweep_type == "r" and not sweep_for_arith:
        partial_sweep_models += [
            "opus-3",
            "sonnet-3-5",
            "sonnet-3-6",
            "sonnet-3-7",
        ]

    # Limited values for partial sweep
    if sweep_type == "r":
        val = 10 if sweep_for_arith else 5
        values = [None, val]
        values_disp = [1, val]
        param_label = "r"
    else:
        val = 100 if sweep_for_arith else 300
        values = [None, val]
        values_disp = [0, val]
        param_label = "f"

    # Run evaluations
    if sweep_type == "r":
        results = await sweep_evaluations(
            partial_sweep_models,
            values,
            sweep_for_arith,
            verbosity=verbosity,
            max_problems=max_problems,
            load_from_output=True,
        )
    else:
        results = await sweep_filler_evaluations(
            partial_sweep_models,
            values,
            sweep_for_arith,
            verbosity=verbosity,
            max_problems=max_problems,
            load_from_output=True,
        )

    # Print summary table
    dataset_name = "Gen-Arithmetic" if sweep_for_arith else "Easy-Comp-Math"
    sweep_type_name = "Repetitions" if sweep_type == "r" else "Filler"
    print("\n" + "=" * 100)
    print(f"PARTIAL SWEEP SUMMARY - {dataset_name} - {sweep_type_name}")
    print("=" * 100)
    print(f"{'Model':<20} | " + " | ".join([f"{param_label}={v:>3}" for v in values_disp]) + " | p-value")
    print("-" * 100)

    # Build markdown table
    markdown_tables = []
    headers = ["Model"] + [f"{param_label}={v}" for v in values_disp] + ["p-value"]
    rows = []

    for model in partial_sweep_models:
        if model in results:
            accuracies = []
            for v in values_disp:
                result = results[model].get(v, 0)
                if isinstance(result, dict):
                    acc = result["accuracy"] * 100
                else:
                    acc = result * 100
                accuracies.append(f"{acc:.1f}%")

            # Calculate significance between first and last value
            try:
                mean_diff, p_value = paired_ttest_on_problems(
                    model, values_disp[0], values_disp[1], sweep_for_arith, sweep_type
                )
                if p_value < 0.001:
                    p_str = f"{p_value:.2e}"
                else:
                    p_str = f"{p_value:.4f}"
            except Exception:
                p_str = "N/A"

            print(f"{model:<20} | " + " | ".join(accuracies) + f" | {p_str}")
            rows.append([model] + accuracies + [p_str])

    markdown_tables.append(generate_markdown_table(f"Partial Sweep - {dataset_name} - {sweep_type_name}", headers, rows))

    # Compute and display time horizon table (only for comp-math, not arithmetic)

    # Generate scatter plot: baseline accuracy vs % accuracy increase, colored by significance
    plot_data = []
    for model in partial_sweep_models:
        if model not in results:
            continue

        # Get baseline and improved accuracies
        baseline_result = results[model].get(values_disp[0], {})
        improved_result = results[model].get(values_disp[1], {})

        if isinstance(baseline_result, dict):
            baseline_acc = baseline_result["accuracy"]
        else:
            baseline_acc = baseline_result

        if isinstance(improved_result, dict):
            improved_acc = improved_result["accuracy"]
        else:
            improved_acc = improved_result

        # Calculate % increase (relative to baseline)
        if baseline_acc > 0:
            pct_increase = (improved_acc - baseline_acc) / baseline_acc * 100
        else:
            pct_increase = 0 if improved_acc == 0 else float("inf")

        # Get p-value
        try:
            _, p_value = paired_ttest_on_problems(
                model, values_disp[0], values_disp[1], sweep_for_arith, sweep_type
            )
        except Exception:
            p_value = 1.0

        plot_data.append({
            "model": model,
            "baseline_acc": baseline_acc * 100,
            "pct_increase": pct_increase,
            "p_value": p_value,
        })

    if plot_data:
        fig, ax = plt.subplots(figsize=(10, 7))

        # Extract data for plotting
        x_vals = [d["baseline_acc"] for d in plot_data]
        y_vals = [d["pct_increase"] for d in plot_data]
        p_vals = [d["p_value"] for d in plot_data]
        labels = [d["model"] for d in plot_data]

        # Color by significance: map p-value to color intensity
        # p < 0.001 -> max saturation, p >= 0.1 -> min saturation
        from matplotlib.colors import LinearSegmentedColormap

        # Create colors based on -log10(p_value), clamped
        colors = []
        for p in p_vals:
            # -log10(0.001) = 3, -log10(0.1) = 1
            # Map to 0-1 range where 0 = not significant (p>=0.1), 1 = highly significant (p<=0.001)
            if p <= 0:
                p = 1e-10
            neg_log_p = -np.log10(p)
            # Clamp between 1 (-log10(0.1)) and 3 (-log10(0.001))
            intensity = (neg_log_p - 1) / 2  # Maps [1, 3] -> [0, 1]
            intensity = np.clip(intensity, 0, 1)
            colors.append(intensity)

        # Use a colormap from gray (not significant) to red (highly significant)
        cmap = plt.cm.Reds
        scatter_colors = [cmap(0.2 + 0.8 * c) for c in colors]  # Offset so even 0 is visible

        # Plot scatter
        scatter = ax.scatter(x_vals, y_vals, c=colors, cmap="Reds", s=150, edgecolors="black", linewidths=1, vmin=0, vmax=1)

        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (x_vals[i], y_vals[i]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Significance (-log₁₀(p), clamped)", fontsize=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["p≥0.1", "p=0.01", "p≤0.001"])

        # Add reference line at y=0
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Labels and title
        ax.set_xlabel(f"Baseline Accuracy (%) [{param_label}={values_disp[0]}]", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"% Accuracy Increase with {param_label}={values_disp[1]}", fontsize=12, fontweight="bold")
        ax.set_title(f"Accuracy Improvement vs Baseline\n{dataset_name} Dataset - {sweep_type_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")


        prefix_file_name = ("arith_" if sweep_for_arith else "") + "partial_sweep"

        plt.tight_layout()
        plot_filename = f"eval_results/{prefix_file_name}_{sweep_type}_accuracy_improvement.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"\nScatter plot saved to: {plot_filename}")
        plt.close()

        # Generate absolute version of the scatter plot
        fig, ax = plt.subplots(figsize=(10, 7))

        # Calculate absolute accuracy increase (in percentage points)
        y_vals_abs = [(d["baseline_acc"] * d["pct_increase"] / 100) if d["pct_increase"] != float("inf") else 0 for d in plot_data]

        # Plot scatter
        scatter = ax.scatter(x_vals, y_vals_abs, c=colors, cmap="Reds", s=150, edgecolors="black", linewidths=1, vmin=0, vmax=1)

        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (x_vals[i], y_vals_abs[i]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Significance (-log₁₀(p), clamped)", fontsize=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["p≥0.1", "p=0.01", "p≤0.001"])

        # Add reference line at y=0
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Labels and title
        ax.set_xlabel(f"Baseline Accuracy (%) [{param_label}={values_disp[0]}]", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Absolute Accuracy Increase (pp) with {param_label}={values_disp[1]}", fontsize=12, fontweight="bold")
        ax.set_title(f"Absolute Accuracy Improvement vs Baseline\n{dataset_name} Dataset - {sweep_type_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plot_filename_abs = f"eval_results/{prefix_file_name}_{sweep_type}_accuracy_improvement_absolute.png"
        plt.savefig(plot_filename_abs, dpi=300, bbox_inches="tight")
        print(f"Scatter plot (absolute) saved to: {plot_filename_abs}")
        plt.close()

        # Generate plot showing overall accuracy with filler/repeat
        fig, ax = plt.subplots(figsize=(10, 7))

        # Get the improved accuracy for each model
        y_vals_overall = []
        for d in plot_data:
            improved_result = results[d["model"]].get(values_disp[1], {})
            if isinstance(improved_result, dict):
                improved_acc = improved_result["accuracy"] * 100
            else:
                improved_acc = improved_result * 100
            y_vals_overall.append(improved_acc)

        # Plot scatter
        scatter = ax.scatter(x_vals, y_vals_overall, c=colors, cmap="Reds", s=150, edgecolors="black", linewidths=1, vmin=0, vmax=1)

        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (x_vals[i], y_vals_overall[i]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Significance (-log₁₀(p), clamped)", fontsize=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["p≥0.1", "p=0.01", "p≤0.001"])

        # Add diagonal reference line (y = x, no improvement)
        min_val = min(min(x_vals), min(y_vals_overall)) - 2
        max_val = max(max(x_vals), max(y_vals_overall)) + 2
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="No improvement (y=x)")

        # Labels and title
        ax.set_xlabel(f"Baseline Accuracy (%) [{param_label}={values_disp[0]}]", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Accuracy (%) with {param_label}={values_disp[1]}", fontsize=12, fontweight="bold")
        ax.set_title(f"Overall Accuracy: Baseline vs With {sweep_type_name}\n{dataset_name} Dataset", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plot_filename_overall = f"eval_results/{prefix_file_name}_{sweep_type}_accuracy_overall.png"
        plt.savefig(plot_filename_overall, dpi=300, bbox_inches="tight")
        print(f"Scatter plot (overall) saved to: {plot_filename_overall}")
        plt.close()

    if sweep_for_arith:
        return markdown_tables

    print("\n" + "=" * 100)
    print(f"TIME HORIZON TABLE - Easy-Comp-Math - {sweep_type_name}")
    print("=" * 100)
    print(
        f"{'Model':<20} | {param_label}={values_disp[0]:>3} horizon | {param_label}={values_disp[1]:>3} horizon | Δ horizon"
    )
    print("-" * 100)

    # Load difficulty ratings once
    difficulty_file = "eval_results/difficulty_ratings.json"
    try:
        difficulty_lookup, _ = load_difficulty_ratings(difficulty_file)
    except FileNotFoundError:
        print(f"Could not find difficulty ratings file: {difficulty_file}")
        difficulty_lookup = None

    # Build time horizon markdown table
    horizon_headers = ["Model", f"{param_label}={values_disp[0]} horizon", f"{param_label}={values_disp[1]} horizon", "Δ horizon"]
    horizon_rows = []

    if difficulty_lookup:
        for model in partial_sweep_models:
            horizons = []
            for v in values_disp:
                data = load_problem_data(model, v, sweep_for_arith, sweep_type)

                # Fit time horizon (suppress printing with verbosity=0)
                result = fit_time_horizon(data["results"], difficulty_lookup, plot_path=None, verbosity=0)
                horizons.append(result["time_horizon_minutes"])

            # Format output
            h1_str = f"{horizons[0]:.1f} min" if horizons[0] is not None else "N/A"
            h2_str = f"{horizons[1]:.1f} min" if horizons[1] is not None else "N/A"
            if horizons[0] is not None and horizons[1] is not None:
                delta = horizons[1] - horizons[0]
                delta_str = f"{delta:+.1f} min"
            else:
                delta_str = "N/A"
            print(f"{model:<20} | {h1_str:>12} | {h2_str:>12} | {delta_str}")
            horizon_rows.append([model, h1_str, h2_str, delta_str])

        markdown_tables.append(generate_markdown_table(f"Time Horizon - Easy-Comp-Math - {sweep_type_name}", horizon_headers, horizon_rows))

        # Generate time horizon doublings plot
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 7))

            # Models to exclude from this plot
            excluded_models = {"gpt-3.5", "haiku-3", "haiku-3-5"}

            # Calculate time horizon doublings for each model
            y_vals_doublings = []
            valid_indices = []
            for i, d in enumerate(plot_data):
                model = d["model"]
                if model in excluded_models:
                    continue
                try:
                    # Load baseline and improved time horizons
                    baseline_data = load_problem_data(model, values_disp[0], sweep_for_arith, sweep_type)
                    improved_data = load_problem_data(model, values_disp[1], sweep_for_arith, sweep_type)

                    baseline_horizon = fit_time_horizon(baseline_data["results"], difficulty_lookup, plot_path=None, verbosity=0)["time_horizon_minutes"]
                    improved_horizon = fit_time_horizon(improved_data["results"], difficulty_lookup, plot_path=None, verbosity=0)["time_horizon_minutes"]

                    if baseline_horizon is not None and improved_horizon is not None and baseline_horizon > 0:
                        multiplier = improved_horizon / baseline_horizon
                        doublings = np.log2(multiplier)
                        y_vals_doublings.append(doublings)
                        valid_indices.append(i)
                except Exception:
                    pass

            if valid_indices:
                x_vals_filtered = [x_vals[i] for i in valid_indices]
                colors_filtered = [colors[i] for i in valid_indices]
                labels_filtered = [labels[i] for i in valid_indices]

                # Plot scatter
                scatter = ax.scatter(x_vals_filtered, y_vals_doublings, c=colors_filtered, cmap="Reds", s=150, edgecolors="black", linewidths=1, vmin=0, vmax=1)

                # Add labels for each point
                for i, label in enumerate(labels_filtered):
                    ax.annotate(label, (x_vals_filtered[i], y_vals_doublings[i]), textcoords="offset points",
                               xytext=(5, 5), fontsize=8, alpha=0.8)

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Significance (-log₁₀(p), clamped)", fontsize=10)
                cbar.set_ticks([0, 0.5, 1])
                cbar.set_ticklabels(["p≥0.1", "p=0.01", "p≤0.001"])

                # Add reference line at y=0 (no change in time horizon)
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

                # Labels and title
                ax.set_xlabel(f"Baseline Accuracy (%) [{param_label}={values_disp[0]}]", fontsize=12, fontweight="bold")
                ax.set_ylabel(f"Additional Time Horizon Doublings with {param_label}={values_disp[1]}", fontsize=12, fontweight="bold")
                ax.set_title(f"Time Horizon Improvement vs Baseline Accuracy\n{dataset_name} Dataset - {sweep_type_name}", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, linestyle="--")

                plt.tight_layout()
                plot_filename_doublings = f"eval_results/{prefix_file_name}_{sweep_type}_time_horizon_doublings.png"
                plt.savefig(plot_filename_doublings, dpi=300, bbox_inches="tight")
                print(f"Time horizon doublings plot saved to: {plot_filename_doublings}")
                plt.close()

    return markdown_tables


def plot_filler_vs_repeat_comparison(sweep_for_arith: bool):
    """
    Generate scatter plots comparing uplift from filler vs uplift from repeats.
    Creates two plots: one for absolute accuracy increase, one for relative accuracy increase.
    """
    dataset_name = "Gen-Arithmetic" if sweep_for_arith else "Easy-Comp-Math"

    # Determine the parameter values used
    if sweep_for_arith:
        r_val = 10
        f_val = 100
    else:
        r_val = 5
        f_val = 300

    # Get list of models that have both r and f data
    # Note: Models added only for sweep_type == "r" in run_partial_sweep are excluded here
    # since they won't have filler data for comparison
    partial_sweep_models = [
        "opus-4-5",
        "opus-4",
        "sonnet-4-5",
        "sonnet-4",
        "haiku-3-5",
        "haiku-3",
        "haiku-4-5",
        "gpt-3.5",
        "gpt-4",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5.1",
        "gpt-5.2",
        "deepseek-v3",
        "qwen3-235b-a22b",
    ]

    # Collect data for each model
    plot_data = []
    for model in partial_sweep_models:
        try:
            # Load repeat data (r=1 and r=val)
            r_baseline_data = load_problem_data(model, 1, sweep_for_arith, "r")
            r_improved_data = load_problem_data(model, r_val, sweep_for_arith, "r")

            # Load filler data (f=0 and f=val)
            f_baseline_data = load_problem_data(model, 0, sweep_for_arith, "f")
            f_improved_data = load_problem_data(model, f_val, sweep_for_arith, "f")

            # Extract accuracies
            r_baseline_acc = r_baseline_data["summary"]["accuracy"]
            r_improved_acc = r_improved_data["summary"]["accuracy"]
            f_baseline_acc = f_baseline_data["summary"]["accuracy"]
            f_improved_acc = f_improved_data["summary"]["accuracy"]

            # Calculate absolute increases (in percentage points)
            r_abs_increase = (r_improved_acc - r_baseline_acc) * 100
            f_abs_increase = (f_improved_acc - f_baseline_acc) * 100

            # Calculate relative increases (%)
            if r_baseline_acc > 0:
                r_rel_increase = (r_improved_acc - r_baseline_acc) / r_baseline_acc * 100
            else:
                r_rel_increase = 0

            if f_baseline_acc > 0:
                f_rel_increase = (f_improved_acc - f_baseline_acc) / f_baseline_acc * 100
            else:
                f_rel_increase = 0

            plot_data.append({
                "model": model,
                "r_abs_increase": r_abs_increase,
                "f_abs_increase": f_abs_increase,
                "r_rel_increase": r_rel_increase,
                "f_rel_increase": f_rel_increase,
                "baseline_acc": r_baseline_acc * 100,  # Use repeat baseline as reference
            })

        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Skipping {model}: {e}")
            continue

    if not plot_data:
        print(f"No data available for filler vs repeat comparison ({dataset_name})")
        return

    # Define model families for coloring
    model_colors = {
        "opus-4-5": "#FF6B6B",
        "opus-4": "#FF6B6B",
        "sonnet-4-5": "#FF6B6B",
        "sonnet-4": "#FF6B6B",
        "haiku-3-5": "#FF6B6B",
        "haiku-3": "#FF6B6B",
        "haiku-4-5": "#FF6B6B",
        "gpt-3.5": "#45B7D1",
        "gpt-4": "#45B7D1",
        "gpt-4o": "#45B7D1",
        "gpt-4.1": "#45B7D1",
        "gpt-5.1": "#45B7D1",
        "gpt-5.2": "#45B7D1",
        "deepseek-v3": "#96CEB4",
        "qwen3-235b-a22b": "#FFEAA7",
    }

    # Create two plots: absolute and relative
    for plot_type in ["absolute", "relative"]:
        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "absolute":
            x_vals = [d["r_abs_increase"] for d in plot_data]
            y_vals = [d["f_abs_increase"] for d in plot_data]
            xlabel = f"Repeat Uplift (pp) [r=1 → r={r_val}]"
            ylabel = f"Filler Uplift (pp) [f=0 → f={f_val}]"
            title = f"Absolute Accuracy Increase: Filler vs Repeat\n{dataset_name} Dataset"
        else:
            x_vals = [d["r_rel_increase"] for d in plot_data]
            y_vals = [d["f_rel_increase"] for d in plot_data]
            xlabel = f"Repeat Uplift (%) [r=1 → r={r_val}]"
            ylabel = f"Filler Uplift (%) [f=0 → f={f_val}]"
            title = f"Relative Accuracy Increase: Filler vs Repeat\n{dataset_name} Dataset"

        labels = [d["model"] for d in plot_data]
        colors = [model_colors.get(d["model"], "#888888") for d in plot_data]

        # Plot scatter points
        for i, (x, y, label, color) in enumerate(zip(x_vals, y_vals, labels, colors)):
            ax.scatter(x, y, c=color, s=150, edgecolors="black", linewidths=1, zorder=5)
            ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, alpha=0.9)

        # Add diagonal line (y = x)
        all_vals = x_vals + y_vals
        min_val = min(all_vals) - 1
        max_val = max(all_vals) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Equal uplift")

        # Add reference lines at 0
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)

        # Shade regions
        ax.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                       alpha=0.1, color="blue", label="Filler better")
        ax.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val],
                       alpha=0.1, color="red", label="Repeat better")

        # Set equal aspect ratio and limits
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")

        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        prefix = "arith_" if sweep_for_arith else ""
        plot_filename = f"eval_results/{prefix}filler_vs_repeat_{plot_type}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {plot_filename}")
        plt.close()


def plot_partial_sweep_bar_chart(sweep_for_arith: bool):
    """
    Create a grouped bar chart showing baseline, repeat, and filler accuracy for each model.
    Includes 95% CI error bars and p-values above repeat/filler bars.
    """
    dataset_name = "Gen-Arithmetic" if sweep_for_arith else "Easy-Comp-Math"

    # Determine the parameter values used
    if sweep_for_arith:
        r_val = 10
        f_val = 100
    else:
        r_val = 5
        f_val = 300

    partial_sweep_models = [
        "opus-4-5",
        "opus-4",
        "sonnet-4-5",
        "sonnet-4",
    ]

    # Add models that only have repeat data (not filler)
    if not sweep_for_arith:
        partial_sweep_models += [
            "sonnet-3-5",
            "sonnet-3-6",
            "sonnet-3-7",
            "opus-3",
        ]

    partial_sweep_models += [
        "haiku-4-5",
        "haiku-3-5",
        "haiku-3",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5",
        "qwen3-235b-a22b",
        "deepseek-v3",
    ]

    # Collect data for each model
    plot_data = []
    for model in partial_sweep_models:
        model_entry = {"model": model}

        # Try to load baseline (r=1) data
        try:
            baseline_data = load_problem_data(model, 1, sweep_for_arith, "r")
            baseline_acc = baseline_data["summary"]["accuracy"]
            baseline_n = baseline_data["summary"]["total"]
            model_entry["baseline_acc"] = baseline_acc
            model_entry["baseline_n"] = baseline_n
        except (FileNotFoundError, KeyError):
            continue  # Skip models without baseline data

        # Try to load repeat (r=val) data
        try:
            repeat_data = load_problem_data(model, r_val, sweep_for_arith, "r")
            repeat_acc = repeat_data["summary"]["accuracy"]
            repeat_n = repeat_data["summary"]["total"]
            model_entry["repeat_acc"] = repeat_acc
            model_entry["repeat_n"] = repeat_n

            # Get p-value for repeat vs baseline
            _, repeat_p = paired_ttest_on_problems(model, 1, r_val, sweep_for_arith, "r")
            model_entry["repeat_p"] = repeat_p
        except (FileNotFoundError, KeyError):
            model_entry["repeat_acc"] = None

        # Try to load filler (f=val) data
        try:
            filler_data = load_problem_data(model, f_val, sweep_for_arith, "f")
            filler_acc = filler_data["summary"]["accuracy"]
            filler_n = filler_data["summary"]["total"]
            model_entry["filler_acc"] = filler_acc
            model_entry["filler_n"] = filler_n

            # Get p-value for filler vs baseline (using f=0 baseline)
            _, filler_p = paired_ttest_on_problems(model, 0, f_val, sweep_for_arith, "f")
            model_entry["filler_p"] = filler_p
        except (FileNotFoundError, KeyError):
            model_entry["filler_acc"] = None

        plot_data.append(model_entry)

    if not plot_data:
        print(f"No data available for bar chart ({dataset_name})")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate bar positions
    n_models = len(plot_data)
    x = np.arange(n_models)
    bar_width = 0.25

    # Calculate 95% CI for each bar
    def calc_ci(acc, n):
        if acc == 0 or acc == 1:
            return 1.96 * np.sqrt((acc * (1 - acc) + 1 / (4 * n)) / n) * 100
        return 1.96 * np.sqrt(acc * (1 - acc) / n) * 100

    # Extract data for plotting
    baseline_accs = [d["baseline_acc"] * 100 for d in plot_data]
    baseline_cis = [calc_ci(d["baseline_acc"], d["baseline_n"]) for d in plot_data]

    repeat_accs = []
    repeat_cis = []
    repeat_ps = []
    for d in plot_data:
        if d.get("repeat_acc") is not None:
            repeat_accs.append(d["repeat_acc"] * 100)
            repeat_cis.append(calc_ci(d["repeat_acc"], d["repeat_n"]))
            repeat_ps.append(d.get("repeat_p", 1.0))
        else:
            repeat_accs.append(None)
            repeat_cis.append(None)
            repeat_ps.append(None)

    filler_accs = []
    filler_cis = []
    filler_ps = []
    for d in plot_data:
        if d.get("filler_acc") is not None:
            filler_accs.append(d["filler_acc"] * 100)
            filler_cis.append(calc_ci(d["filler_acc"], d["filler_n"]))
            filler_ps.append(d.get("filler_p", 1.0))
        else:
            filler_accs.append(None)
            filler_cis.append(None)
            filler_ps.append(None)

    # Plot baseline bars
    bars_baseline = ax.bar(x - bar_width, baseline_accs, bar_width, yerr=baseline_cis,
                           label="Baseline (r=1)", color="#4A90A4", capsize=3, error_kw={"elinewidth": 1})

    # Plot repeat bars (only where data exists)
    for i, (acc, ci) in enumerate(zip(repeat_accs, repeat_cis)):
        if acc is not None:
            ax.bar(x[i], acc, bar_width, yerr=ci, color="#E07B53", capsize=3, error_kw={"elinewidth": 1},
                   label=f"Repeat (r={r_val})" if i == 0 else "")

    # Plot filler bars (only where data exists)
    first_filler = True
    for i, (acc, ci) in enumerate(zip(filler_accs, filler_cis)):
        if acc is not None:
            ax.bar(x[i] + bar_width, acc, bar_width, yerr=ci, color="#7FB069", capsize=3, error_kw={"elinewidth": 1},
                   label=f"Filler (f={f_val})" if first_filler else "")
            first_filler = False

    # Format p-value for display
    def format_p(p):
        if p < 0.001:
            return f"p<.001"
        elif p < 0.01:
            return f"p={p:.3f}"
        elif p < 0.05:
            return f"p={p:.2f}"
        else:
            return f"p={p:.2f}"

    # Add p-values above repeat and filler bars
    for i in range(n_models):
        # Repeat p-value
        if repeat_accs[i] is not None:
            p_text = format_p(repeat_ps[i])
            bar_height = repeat_accs[i] + repeat_cis[i]
            ax.text(x[i], bar_height + 1, p_text, ha="center", va="bottom", fontsize=7, rotation=90)

        # Filler p-value
        if filler_accs[i] is not None:
            p_text = format_p(filler_ps[i])
            bar_height = filler_accs[i] + filler_cis[i]
            ax.text(x[i] + bar_width, bar_height + 1, p_text, ha="center", va="bottom", fontsize=7, rotation=90)

    # Customize plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Model Accuracy: Baseline vs Repeat vs Filler\n{dataset_name} Dataset (95% CI, paired t-test p-values)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d["model"] for d in plot_data], rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    prefix = "arith_" if sweep_for_arith else ""
    plot_filename = f"eval_results/{prefix}partial_sweep_bar_chart.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {plot_filename}")
    plt.close()


async def main():
    """Main function to run sweep and create plot."""
    verbosity = 0
    all_markdown_tables = ["# Sweep Results Summary\n"]

    for sweep_for_arith in [True, False]:
        # Sweep type: "r" for problem repetitions, "f" for filler tokens
        for sweep_type in ["r", "f"]:
            md_table = await run_sweep(sweep_type, sweep_for_arith, verbosity=verbosity)
            all_markdown_tables.append(md_table)

    for sweep_for_arith in [True, False]:
        for sweep_type in ["r", "f"]:
            md_tables = await run_partial_sweep(sweep_type=sweep_type, sweep_for_arith=sweep_for_arith, verbosity=verbosity)
            all_markdown_tables.extend(md_tables)

    # Generate filler vs repeat comparison plots
    for sweep_for_arith in [True, False]:
        plot_filler_vs_repeat_comparison(sweep_for_arith)

    # Generate bar charts comparing baseline vs repeat vs filler
    for sweep_for_arith in [True, False]:
        plot_partial_sweep_bar_chart(sweep_for_arith)

    # Write all markdown tables to file
    markdown_output = "\n".join(all_markdown_tables)
    with open("eval_results/summary_tables.md", "w") as f:
        f.write(markdown_output)
    print(f"\nMarkdown summary saved to: eval_results/summary_tables.md")


if __name__ == "__main__":
    asyncio.run(main())
