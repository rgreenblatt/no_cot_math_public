#!/usr/bin/env python3
"""
Analyze no-reasoning results, focusing on hardest problems the model gets correct.
Uses difficulty ratings from rate_difficulty.py instead of problem index.
"""

import json
import argparse
import sys
import math
import random

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function


def load_results(filepath):
    """Load evaluation results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_difficulty_ratings(filepath):
    """Load difficulty ratings and create a lookup by problem text."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create lookup by problem text (exact match) - includes both difficulty and solve time
    ratings_by_problem = {}
    for result in data["results"]:
        problem_text = result["problem"]
        ratings_by_problem[problem_text] = {
            "difficulty_rating": result.get("difficulty_rating"),
            "solve_time_minutes": result.get("solve_time_minutes"),
        }

    return ratings_by_problem, data


def verify_problems_match(eval_results, difficulty_data):
    """
    Verify that problems in eval results match those in difficulty ratings.
    Returns True if all problems match, False otherwise.
    """
    eval_problems = {r["problem"] for r in eval_results}
    difficulty_problems = {r["problem"] for r in difficulty_data["results"]}

    missing_in_difficulty = eval_problems - difficulty_problems
    missing_in_eval = difficulty_problems - eval_problems

    if missing_in_difficulty:
        print(f"ERROR: {len(missing_in_difficulty)} problems in eval results not found in difficulty ratings:")
        for p in list(missing_in_difficulty)[:3]:
            print(f"  - {p[:100]}...")
        return False

    if missing_in_eval:
        print(f"WARNING: {len(missing_in_eval)} problems in difficulty ratings not found in eval results")

    return True


def fit_time_horizon(input_results, difficulty_lookup, plot_path=None, verbosity=2):
    """
    Fit a logistic model to determine the 50% reliability time horizon.

    Uses the METR methodology:
    p_success = sigmoid((log2(h) - log2(t)) * beta)

    Where:
    - h is the time horizon (time at which model has 50% success)
    - t is the human solve time for the task
    - beta is the slope parameter

    Args:
        input_results: List of evaluation results
        difficulty_lookup: Dict mapping problem text to difficulty info
        plot_path: If provided, save a plot to this path
        verbosity: 0 = silent, 1 = normal output (default)

    Returns:
        dict with fitted parameters and statistics
    """
    # Collect data points: (solve_time, success)
    data_points = []
    for r in input_results:
        info = difficulty_lookup.get(r["problem"], {})
        solve_time = info.get("solve_time_minutes")
        if solve_time is not None and solve_time > 0:
            data_points.append(
                {
                    "solve_time": solve_time,
                    "log2_time": np.log2(solve_time),
                    "success": 1 if r["is_correct"] else 0,
                }
            )

    if len(data_points) < 10:
        if verbosity >= 1:
            print("WARNING: Not enough data points with solve times to fit time horizon model")
        return None

    # Convert to arrays
    log2_times = np.array([d["log2_time"] for d in data_points])
    successes = np.array([d["success"] for d in data_points])
    times = np.array([d["solve_time"] for d in data_points])

    # Fit logistic model: p = sigmoid((log2_h - log2_t) * beta)
    # Reparameterize: p = sigmoid(a + b * log2_t) where a = log2_h * beta, b = -beta
    # Then: log2_h = -a/b, beta = -b

    def neg_log_likelihood(params):
        a, b = params
        logits = a + b * log2_times
        # Clip to avoid numerical issues
        logits = np.clip(logits, -500, 500)
        probs = expit(logits)
        # Binary cross-entropy
        eps = 1e-10
        probs = np.clip(probs, eps, 1 - eps)
        ll = successes * np.log(probs) + (1 - successes) * np.log(1 - probs)
        return -np.sum(ll)

    # Initial guess
    x0 = [0.0, -1.0]  # Expect negative slope (higher time = lower success)

    # Fit
    result = minimize(neg_log_likelihood, x0, method="Nelder-Mead")
    a, b = result.x

    # Extract parameters
    # From a + b * log2_t = 0 at 50%, we get log2_h = -a/b
    if abs(b) < 1e-10:
        if verbosity >= 1:
            print("WARNING: Slope too close to zero, cannot determine time horizon")
        return None

    log2_h = -a / b
    time_horizon = 2**log2_h
    beta = -b

    # Calculate R² (McFadden's pseudo R²)
    null_ll = -neg_log_likelihood([np.log(successes.mean() / (1 - successes.mean() + 1e-10)), 0])
    fitted_ll = -neg_log_likelihood([a, b])
    pseudo_r2 = 1 - (fitted_ll / null_ll) if null_ll != 0 else 0

    # Print results
    if verbosity >= 2:
        print(f"\n{'='*60}")
        print("50% RELIABILITY TIME HORIZON (METR-style)")
        print(f"{'='*60}")
        print(f"Data points: {len(data_points)}")
        print(f"Success rate: {successes.mean():.1%}")
        print(f"Time range: {times.min():.1f} - {times.max():.1f} minutes")
        print(f"\nFitted parameters:")
        print(f"  Time horizon (50% reliability): {time_horizon:.1f} minutes")
        print(f"  Slope (beta): {beta:.3f}")
        print(f"  Pseudo R²: {pseudo_r2:.3f}")
        print(f"\nInterpretation:")
        print(f"  The model has ~50% chance of solving problems that take")
        print(f"  a median AIME qualifier {time_horizon:.1f} minutes to solve.")

    # Generate plot if requested
    if plot_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))

            # Generate smooth curve
            t_range = np.linspace(max(0.5, times.min() * 0.5), times.max() * 1.5, 200)
            log2_t_range = np.log2(t_range)
            p_range = expit(a + b * log2_t_range)

            # Add jittered data points
            jitter = np.random.normal(0, 0.02, len(successes))

            # Binned success rates
            n_bins = 15
            bin_edges = np.percentile(times, np.linspace(0, 100, n_bins + 1))
            bin_centers = []
            bin_rates = []
            bin_counts = []
            for i in range(n_bins):
                mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
                if i == n_bins - 1:
                    mask = (times >= bin_edges[i]) & (times <= bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_centers.append(times[mask].mean())
                    bin_rates.append(successes[mask].mean())
                    bin_counts.append(mask.sum())

            # Log scale plot
            ax.plot(t_range, p_range, "b-", linewidth=2, label="Fitted sigmoid")
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
            ax.axvline(
                x=time_horizon, color="red", linestyle="--", alpha=0.7, label=f"Time horizon: {time_horizon:.1f} min"
            )
            ax.scatter(times, successes + jitter, alpha=0.3, s=20, c="green")
            ax.scatter(
                bin_centers,
                bin_rates,
                s=[c * 3 for c in bin_counts],
                c="orange",
                edgecolors="black",
                zorder=5,
                label="Binned success rate",
            )

            ax.set_xlabel("Estimated solve time (minutes, log scale)")
            ax.set_ylabel("Model success probability")
            ax.set_title("50% Reliability Time Horizon (Log Scale)")
            ax.set_xscale("log")
            ax.legend(loc="upper right")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            if verbosity >= 1:
                print(f"\nPlot saved to: {plot_path}")
            plt.close()

        except ImportError:
            if verbosity >= 1:
                print("\nWARNING: matplotlib not installed, skipping plot")

    return {
        "time_horizon_minutes": time_horizon,
        "beta": beta,
        "pseudo_r2": pseudo_r2,
        "n_data_points": len(data_points),
        "success_rate": successes.mean(),
        "a": a,
        "b": b,
    }


def plot_solve_rate_by_difficulty(input_results, difficulty_lookup, plot_path=None, verbosity=1):
    """
    Plot solve rate by difficulty level as a bar chart.

    Args:
        input_results: List of evaluation results
        difficulty_lookup: Dict mapping problem text to difficulty info
        plot_path: If provided, save the plot to this path
        verbosity: 0 = silent, 1 = normal output
    """
    # Augment results with difficulty
    augmented = []
    for r in input_results:
        info = difficulty_lookup.get(r["problem"], {})
        difficulty = info.get("difficulty_rating")
        if difficulty is not None:
            augmented.append(
                {
                    "is_correct": r["is_correct"],
                    "difficulty_rating": difficulty,
                }
            )

    if not augmented:
        if verbosity >= 1:
            print("WARNING: No problems with difficulty ratings found")
        return None

    # Calculate solve rate by difficulty
    difficulties = []
    solve_rates = []
    counts = []

    for diff in range(1, 11):
        problems_at_diff = [r for r in augmented if r["difficulty_rating"] == diff]
        total = len(problems_at_diff)
        if total > 0:
            correct = sum(1 for r in problems_at_diff if r["is_correct"])
            rate = correct / total
            difficulties.append(diff)
            solve_rates.append(rate)
            counts.append(total)

    if not difficulties:
        if verbosity >= 1:
            print("WARNING: No valid difficulty data to plot")
        return None

    # Generate plot
    if plot_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create bar chart
            bars = ax.bar(difficulties, solve_rates, color='steelblue', edgecolor='black', alpha=0.8)

            # Add count labels on top of bars
            for bar, count, rate in zip(bars, counts, solve_rates):
                height = bar.get_height()
                ax.annotate(f'n={count}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            # Add percentage labels inside bars
            for bar, rate in zip(bars, solve_rates):
                height = bar.get_height()
                if height > 0.05:  # Only show label if bar is tall enough
                    ax.annotate(f'{rate:.0%}',
                                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

            ax.set_xlabel('Difficulty Rating', fontsize=12)
            ax.set_ylabel('Solve Rate', fontsize=12)
            ax.set_title('Model Solve Rate by Problem Difficulty', fontsize=14)
            ax.set_xticks(range(1, 11))
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0.5, 10.5)

            # Add horizontal grid lines
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)

            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

            # Add overall accuracy line
            overall_correct = sum(1 for r in augmented if r["is_correct"])
            overall_rate = overall_correct / len(augmented)
            ax.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2,
                       label=f'Overall accuracy: {overall_rate:.1%}')
            ax.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            if verbosity >= 1:
                print(f"Solve rate by difficulty plot saved to: {plot_path}")
            plt.close()

        except ImportError:
            if verbosity >= 1:
                print("WARNING: matplotlib not installed, skipping plot")

    return {
        "difficulties": difficulties,
        "solve_rates": solve_rates,
        "counts": counts,
    }


def print_solve_rate_tables(input_results, difficulty_lookup):
    """Print tables showing solve rate by difficulty and by solve time."""

    # Augment results with difficulty and solve time
    augmented = []
    for r in input_results:
        info = difficulty_lookup.get(r["problem"], {})
        difficulty = info.get("difficulty_rating")
        solve_time = info.get("solve_time_minutes")
        if difficulty is not None:
            augmented.append(
                {
                    "is_correct": r["is_correct"],
                    "difficulty_rating": difficulty,
                    "solve_time_minutes": solve_time,
                }
            )

    # Table 1: Solve rate by difficulty
    print(f"\n{'='*60}")
    print("SOLVE RATE BY DIFFICULTY")
    print(f"{'='*60}")
    print(f"{'Difficulty':<12} {'Correct':<10} {'Total':<10} {'Solve Rate':<12}")
    print(f"{'-'*44}")

    for diff in range(1, 11):
        problems_at_diff = [r for r in augmented if r["difficulty_rating"] == diff]
        total = len(problems_at_diff)
        if total > 0:
            correct = sum(1 for r in problems_at_diff if r["is_correct"])
            rate = correct / total
            bar = "█" * int(rate * 20)
            print(f"{diff:<12} {correct:<10} {total:<10} {rate:>6.1%} {bar}")

    # Overall
    total_all = len(augmented)
    correct_all = sum(1 for r in augmented if r["is_correct"])
    rate_all = correct_all / total_all if total_all > 0 else 0
    print(f"{'-'*44}")
    print(f"{'Overall':<12} {correct_all:<10} {total_all:<10} {rate_all:>6.1%}")

    # Table 2: Solve rate by solve time (buckets)
    print(f"\n{'='*60}")
    print("SOLVE RATE BY ESTIMATED SOLVE TIME")
    print(f"{'='*60}")
    print(f"{'Time (min)':<12} {'Correct':<10} {'Total':<10} {'Solve Rate':<12}")
    print(f"{'-'*44}")

    time_buckets = [
        (0, 0.5),
        (0.5, 1),
        (1, 3),
        (3, 5),
        (5, 10),
        (10, 20),
        (20, 40),
        (40, float("inf")),
    ]
    bucket_labels = [f"{low}-{high}" if high != float("inf") else f"{low}+" for low, high in time_buckets]

    for label, (low, high) in zip(bucket_labels, time_buckets):
        problems_in_bucket = [
            r for r in augmented if r["solve_time_minutes"] is not None and low <= r["solve_time_minutes"] < high
        ]
        total = len(problems_in_bucket)
        if total > 0:
            correct = sum(1 for r in problems_in_bucket if r["is_correct"])
            rate = correct / total
            bar = "█" * int(rate * 20)
            print(f"{label:<12} {correct:<10} {total:<10} {rate:>6.1%} {bar}")

    # Problems with missing solve time
    missing_time = [r for r in augmented if r["solve_time_minutes"] is None]
    if missing_time:
        print(f"{'(no time)':<12} {'-':<10} {len(missing_time):<10} {'N/A':<12}")

    print()


def analyze_hardest_correct(input_file, difficulty_file, with_reasoning_file=None, top_n=10, plot_path=None, difficulty_plot_path=None):
    """
    Analyze the hardest problems that the model gets correct without reasoning.

    Args:
        input_file: Path to no-reasoning results JSON
        difficulty_file: Path to difficulty ratings JSON from rate_difficulty.py
        with_reasoning_file: Path to with-reasoning results JSON (optional)
        top_n: Number of hardest correct problems to show
        plot_path: Path to save the time horizon plot (optional)
        difficulty_plot_path: Path to save the solve rate by difficulty plot (optional)
    """
    # Load no-reasoning results
    input_data = load_results(input_file)
    input_results = input_data["results"]

    # Load difficulty ratings
    try:
        difficulty_lookup, difficulty_data = load_difficulty_ratings(difficulty_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find difficulty ratings file: {difficulty_file}")
        print("Run rate_difficulty.py first to generate difficulty ratings.")
        sys.exit(1)

    # Verify problems match exactly
    if not verify_problems_match(input_results, difficulty_data):
        print("ERROR: Problems do not match between eval results and difficulty ratings.")
        sys.exit(1)

    print(f"Verified: All {len(input_results)} problems match between eval and difficulty files.\n")

    # Print solve rate tables
    print_solve_rate_tables(input_results, difficulty_lookup)

    # Plot solve rate by difficulty
    plot_solve_rate_by_difficulty(input_results, difficulty_lookup, plot_path=difficulty_plot_path)

    # Fit and plot time horizon
    fit_time_horizon(input_results, difficulty_lookup, plot_path=plot_path)

    # Load with-reasoning results if provided
    with_reasoning_results = None
    if with_reasoning_file:
        try:
            with_reasoning_data = load_results(with_reasoning_file)
            with_reasoning_results = {
                r["problem"]: r for r in with_reasoning_data["results"]  # Use problem text as key for matching
            }
        except FileNotFoundError:
            print(f"Warning: Could not find {with_reasoning_file}, skipping reasoning comparison\n")

    # Filter for correct answers and add difficulty rating
    correct_problems = []
    for r in input_results:
        if r["is_correct"]:
            info = difficulty_lookup.get(r["problem"], {})
            difficulty = info.get("difficulty_rating")
            if difficulty is not None:
                r["difficulty_rating"] = difficulty
                r["solve_time_minutes"] = info.get("solve_time_minutes")
                correct_problems.append(r)

    # Sort by difficulty rating (higher = harder)
    correct_problems.sort(key=lambda x: x["difficulty_rating"], reverse=True)

    # Get top N hardest
    hardest_correct = correct_problems[:top_n]

    # Print summary
    print(f"=" * 80)
    print(f"NO-REASONING ANALYSIS")
    print(f"=" * 80)
    print(f"Total problems evaluated: {input_data['summary']['total']}")
    print(f"Correct: {input_data['summary']['correct']}")
    print(f"Accuracy: {input_data['summary']['accuracy']:.2%}")
    print(f"\nShowing top {len(hardest_correct)} hardest problems that were answered CORRECTLY:")
    print(f"=" * 80)
    print()

    # Print each hardest correct problem
    for i, result in enumerate(hardest_correct, 1):
        difficulty = result["difficulty_rating"]
        solve_time = result.get("solve_time_minutes")
        time_str = f"{solve_time:.0f} min" if solve_time else "N/A"
        print(
            f"#{i} - Difficulty: {difficulty}/10, Est. solve time: {time_str} (Category: {result['category']}, Problem #: {result['problem_number']})"
        )
        print(f"Round: {result['round']}")
        print(f"-" * 80)
        print(f"Problem:\n{result['problem']}")
        print(f"\nCorrect Answer: {result['correct_answer']}")
        print(f"Model Response (no reasoning): {result['response']}")
        print(f"Predicted Answer: {result['predicted_answer']}")

        # Show with-reasoning version if available (match by problem text)
        problem_text = result["problem"]
        if with_reasoning_results and problem_text in with_reasoning_results:
            reasoning_result = with_reasoning_results[problem_text]
            print(f"\n--- WITH REASONING VERSION ---")
            print(f"Correct (with reasoning): {reasoning_result['is_correct']}")
            print(f"Predicted (with reasoning): {reasoning_result['predicted_answer']}")

            # Show thinking if available (truncated)
            if "thinking" in reasoning_result and reasoning_result["thinking"]:
                thinking = reasoning_result["thinking"]
                if len(thinking) > 1000:
                    print(f"\nThinking (first 1000 chars):\n{thinking[:1000]}...")
                else:
                    print(f"\nThinking:\n{thinking}")

            # # Show response
            # if 'response' in reasoning_result:
            #     response = reasoning_result['response']
            #     if len(response) > 300:
            #         print(f"\nResponse (first 300 chars):\n{response[:300]}...")
            #     else:
            #         print(f"\nResponse:\n{response}")

        print(f"\n{'='*80}\n")



def show_error_analysis(input_file):
    """Show analysis of incorrect answers."""
    data = load_results(input_file)
    results = data["results"]

    incorrect = [r for r in results if not r["is_correct"]]

    print(f"=" * 80)
    print(f"ERROR ANALYSIS")
    print(f"=" * 80)
    print(f"Total incorrect: {len(incorrect)}")

    # Show distribution by category
    by_category = {}
    for r in incorrect:
        cat = r["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"\nIncorrect by category:")
    for cat, count in sorted(by_category.items()):
        total_in_cat = len([r for r in results if r["category"] == cat])
        print(f"  {cat}: {count}/{total_in_cat} ({count/total_in_cat*100:.1f}% error rate)")

    # Show some examples of errors
    print(f"\nExample errors (first 5):")
    for r in incorrect[:5]:
        print(f"\nProblem {r['problem_index']}: {r['problem'][:150]}...")
        print(f"  Correct: {r['correct_answer']}, Predicted: {r['predicted_answer']}")
        print(f"  Response: {r['response']}")


# Time bucket definitions (shared between functions)
TIME_BUCKETS = [
    (0, 0.5, "0-0.5"),
    (0.5, 1, "0.5-1"),
    (1, 3, "1-3"),
    (3, 5, "3-5"),
    (5, 10, "5-10"),
    (10, 20, "10-20"),
    (20, 40, "20-40"),
    (40, float("inf"), "40+"),
]


def parse_time_bucket(bucket_str):
    """
    Parse a time bucket string and return (low, high) bounds.
    Accepts formats like: "0-0.5", "0.5-1", "40+", "40-inf"
    """
    bucket_str = bucket_str.strip().lower()

    # Check for "+" suffix (e.g., "40+")
    if bucket_str.endswith("+"):
        low = float(bucket_str[:-1])
        return (low, float("inf"))

    # Check for "-inf" suffix
    if bucket_str.endswith("-inf"):
        low = float(bucket_str[:-4])
        return (low, float("inf"))

    # Standard range format "low-high"
    if "-" in bucket_str:
        parts = bucket_str.split("-")
        if len(parts) == 2:
            low = float(parts[0])
            high = float(parts[1])
            return (low, high)

    raise ValueError(f"Invalid time bucket format: '{bucket_str}'. Use formats like '0-0.5', '5-10', or '40+'")


def show_random_problems(
    input_file, difficulty_file, difficulty_level=None, time_bucket=None, num_samples=5, with_reasoning_file=None, only_show_correct=False
):
    """
    Show random problems from a particular difficulty level or time bucket.

    Args:
        input_file: Path to no-reasoning results JSON
        difficulty_file: Path to difficulty ratings JSON
        difficulty_level: Difficulty level (1-10) to filter by, or None for all
        time_bucket: Time bucket string (e.g., "5-10", "40+") to filter by, or None for all
        num_samples: Number of random problems to show
        with_reasoning_file: Path to with-reasoning results JSON (optional)
    """
    # Load data
    input_data = load_results(input_file)
    input_results = input_data["results"]

    try:
        difficulty_lookup, difficulty_data = load_difficulty_ratings(difficulty_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find difficulty ratings file: {difficulty_file}")
        sys.exit(1)

    # Load with-reasoning results if provided
    with_reasoning_results = None
    if with_reasoning_file:
        try:
            with_reasoning_data = load_results(with_reasoning_file)
            with_reasoning_results = {r["problem"]: r for r in with_reasoning_data["results"]}
        except FileNotFoundError:
            print(f"Warning: Could not find {with_reasoning_file}")

    # Augment results with difficulty and solve time
    augmented = []
    for r in input_results:
        info = difficulty_lookup.get(r["problem"], {})
        difficulty = info.get("difficulty_rating")
        solve_time = info.get("solve_time_minutes")
        if difficulty is not None:
            r_copy = r.copy()
            r_copy["difficulty_rating"] = difficulty
            r_copy["solve_time_minutes"] = solve_time
            augmented.append(r_copy)

    # Filter by difficulty level
    if difficulty_level is not None:
        augmented = [r for r in augmented if r["difficulty_rating"] == difficulty_level]

    # Filter by time bucket
    if time_bucket is not None:
        low, high = parse_time_bucket(time_bucket)
        augmented = [
            r for r in augmented if r["solve_time_minutes"] is not None and low <= r["solve_time_minutes"] < high
        ]

    if only_show_correct:
        augmented = [r for r in augmented if r["is_correct"]]

    if not augmented:
        print("No problems found matching the specified criteria.")
        return


    print(f"\nTotal problems matching criteria: {len(augmented)}")

    if not only_show_correct:
        accuracy_in_bucket = sum(1 for r in augmented if r["is_correct"]) / len(augmented)
        print(f"Accuracy in this set: {accuracy_in_bucket:.2%}\n")

    # Random sample
    num_to_show = min(num_samples, len(augmented))
    sampled = random.sample(augmented, num_to_show)

    # Print header
    print(f"=" * 80)
    print(f"RANDOM PROBLEMS")
    filters = []
    if difficulty_level is not None:
        filters.append(f"difficulty={difficulty_level}")
    if time_bucket is not None:
        filters.append(f"time_bucket={time_bucket}")
    filter_str = ", ".join(filters) if filters else "no filters"
    print(f"Filters: {filter_str}")
    print(f"Showing {num_to_show} of {len(augmented)} matching problems")
    print(f"=" * 80)

    # Print each problem
    for i, result in enumerate(sampled, 1):
        difficulty = result["difficulty_rating"]
        solve_time = result.get("solve_time_minutes")
        time_str = f"{solve_time:.1f} min" if solve_time else "N/A"
        correct_str = "✓ CORRECT" if result["is_correct"] else "✗ INCORRECT"

        print(f"\n#{i} - Difficulty: {difficulty}/10, Est. solve time: {time_str} [{correct_str}]")
        print(f"Category: {result['category']}, Problem #: {result['problem_number']}, Round: {result['round']}")
        print(f"-" * 80)
        print(f"Problem:\n{result['problem']}")
        print(f"\nCorrect Answer: {result['correct_answer']}")
        print(f"Model Response (no reasoning): {result['response']}")
        print(f"Predicted Answer: {result['predicted_answer']}")

        # Show with-reasoning version if available
        problem_text = result["problem"]
        if with_reasoning_results and problem_text in with_reasoning_results:
            reasoning_result = with_reasoning_results[problem_text]
            print(f"\n--- WITH REASONING VERSION ---")
            print(f"Correct (with reasoning): {reasoning_result['is_correct']}")
            print(f"Predicted (with reasoning): {reasoning_result['predicted_answer']}")

            if "thinking" in reasoning_result and reasoning_result["thinking"]:
                thinking = reasoning_result["thinking"]
                if len(thinking) > 1000:
                    print(f"\nThinking (first 1000 chars):\n{thinking[:1000]}...")
                else:
                    print(f"\nThinking:\n{thinking}")

        print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze (no-reasoning) evaluation results")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="eval_results/eval_results_no_reasoning.json",
        help="Input results file (default: eval_results/eval_results_no_reasoning.json)",
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        default="eval_results/difficulty_ratings.json",
        help="Difficulty ratings file from rate_difficulty.py (default: eval_results/difficulty_ratings.json)",
    )
    parser.add_argument(
        "--with-reasoning",
        "-r",
        type=str,
        default="eval_results/eval_results.json",
        help="With-reasoning results file for comparison (default: eval_results/eval_results.json)",
    )
    parser.add_argument(
        "--top", "-t", type=int, default=0, help="Number of hardest correct problems to show (default: 10)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Show comparison between input file and with-reasoning"
    )
    parser.add_argument("--errors", action="store_true", help="Show error analysis")
    parser.add_argument(
        "--plot",
        "-p",
        type=str,
        default="eval_results/time_horizon_plot.png",
        help="Path to save time horizon plot (default: eval_results/time_horizon_plot.png)",
    )
    parser.add_argument(
        "--difficulty-plot",
        type=str,
        default="eval_results/solve_rate_by_difficulty.png",
        help="Path to save solve rate by difficulty plot (default: eval_results/solve_rate_by_difficulty.png)",
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Show N random problems (use with --filter-difficulty and/or --filter-time)",
    )
    parser.add_argument(
        "--filter-difficulty",
        type=int,
        choices=range(1, 11),
        metavar="1-10",
        help="Filter random problems by difficulty level (1-10)",
    )
    parser.add_argument(
        "--filter-time",
        type=str,
        metavar="BUCKET",
        help="Filter random problems by time bucket (e.g., '0-0.5', '5-10', '40+')",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="When showing random problems, only show those answered correctly",
    )

    args = parser.parse_args()

    # If --random is specified, show random problems and exit
    if args.random is not None:
        show_random_problems(
            args.input,
            args.difficulty,
            difficulty_level=args.filter_difficulty,
            time_bucket=args.filter_time,
            num_samples=args.random,
            with_reasoning_file=args.with_reasoning,
            only_show_correct=args.only_correct,
        )
        sys.exit(0)

    # Main analysis: hardest correct problems
    analyze_hardest_correct(args.input, args.difficulty, args.with_reasoning, args.top, args.plot, args.difficulty_plot)

    # # Optional: error analysis
    # if args.errors:
    #     show_error_analysis(args.input)
