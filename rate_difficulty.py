#!/usr/bin/env python3
"""
Rate math problems by difficulty using Claude with extended thinking.
"""

import json
import os
import asyncio
import re
from anthropic import AsyncAnthropic
import matplotlib.pyplot as plt
from response_cache import ResponseCache

if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

# Initialize Anthropic client
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize cache
response_cache = ResponseCache("caches/difficulty_cache.json")


def load_problems(filepath):
    """Load problems from JSONL file."""
    problems = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def extract_rating_and_time(text):
    """Extract the difficulty rating (1-10) and solve time (minutes) from the response."""
    rating = None
    solve_time = None

    # Extract difficulty rating
    rating_patterns = [
        r"difficulty[:\s]+(\d+)",
        r"rating[:\s]+(\d+)",
    ]
    for pattern in rating_patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = int(match.group(1))
            if 1 <= val <= 10:
                rating = val
                break

    # Extract solve time (in minutes)
    time_patterns = [
        r"solve[_ ]?time[:\s]+(\d+(?:\.\d+)?)",
        r"time[:\s]+(\d+(?:\.\d+)?)\s*(?:min|minutes)?",
        r"(\d+(?:\.\d+)?)\s*(?:min|minutes)",
    ]
    for pattern in time_patterns:
        match = re.search(pattern, text.lower())
        if match:
            solve_time = float(match.group(1))
            break

    return rating, solve_time


async def rate_problem(problem, semaphore, problem_index, model="claude-opus-4-5-20251101"):
    """
    Send a problem to Claude to rate its difficulty.

    Args:
        problem: dict with 'problem' key
        semaphore: asyncio.Semaphore to limit concurrency
        problem_index: index of the problem for logging
        model: Claude model to use

    Returns:
        dict with rating results
    """
    prompt = f"""Rate the difficulty of the following math problem and estimate how long it would take to solve.

DIFFICULTY RATING (1-10 scale):
- 1 is an easy grade school math problem (basic arithmetic, simple word problems)
- 3 is a challenging middle school or early high school problem—think algebra, basic geometry, or problems from competitions like AMC 8 or early AMC 10
- 5 is a solid high school competition problem, like a mid-range AMC 10/12 question
- 8 is a typical AIME problem
- 10 is harder than the hardest AIME problem (approaching olympiad level)

SOLVE TIME: Estimate the median time (in minutes) it would take for a typical AIME qualifier to solve this problem. An "AIME qualifier" is a high school student who scored well enough on the AMC 10/12 to qualify for the AIME. Consider that these students are strong but not necessarily top competitors. Use fractional minutes when appropriate (e.g., 0.25, 0.5, 1.5, 2.5) - very easy problems may take less than a minute.

Consider factors like:
- Mathematical concepts required
- Number of steps to solve
- Creativity or insight needed
- Computational complexity

Problem: {problem['problem']}

After thinking about the problem, respond with exactly two lines:
Difficulty: X
Solve_time: Y

Where X is your difficulty rating (1-10) and Y is your estimated solve time in minutes."""

    # Define API call parameters
    max_tokens = 6_000
    thinking_config = {"type": "enabled", "budget_tokens": 4_000}
    messages = [{"role": "user", "content": prompt}]

    # Check cache first
    cache_key = {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": thinking_config,
        "messages": messages,
    }
    cached_response = await response_cache.get(cache_key)

    async with semaphore:
        try:
            thinking_text = ""
            response_text = ""

            if cached_response:
                print(
                    f"[CACHED] Problem {problem_index + 1}: {problem.get('category')}, {problem.get('problem_number')}"
                )
                thinking_text = cached_response.get("thinking", "")
                response_text = cached_response.get("response", "")
            else:
                print(f"Rating problem {problem_index + 1}: {problem.get('category')}, {problem.get('problem_number')}")

                try:
                    response = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            thinking=thinking_config,
                            messages=messages,
                            timeout=120.0,
                        ),
                        timeout=180.0,
                    )

                    for block in response.content:
                        if block.type == "thinking":
                            thinking_text = block.thinking
                        elif block.type == "text":
                            response_text = block.text

                    # Cache the response
                    response_data = {
                        "thinking": thinking_text,
                        "response": response_text,
                    }
                    await response_cache.set(cache_key, response_data)

                except asyncio.TimeoutError:
                    print(f"TIMEOUT on problem {problem_index + 1}")
                    raise Exception("API call timed out")

            # Extract rating and solve time
            rating, solve_time = extract_rating_and_time(response_text)

            result = {
                "problem_index": problem_index,
                "problem_number": problem.get("problem_number", "N/A"),
                "category": problem.get("category", "N/A"),
                "round": problem.get("round", "N/A"),
                "difficulty_rating": rating,
                "solve_time_minutes": solve_time,
                "response": response_text,
                "thinking": thinking_text,
                "problem": problem["problem"],
                "answer": problem.get("answer"),
                "cached": cached_response is not None,
            }

            cache_status = "[CACHED]" if cached_response else ""
            time_str = f"{solve_time:.1f}min" if solve_time else "N/A"
            print(f"Completed problem {problem_index + 1}: Difficulty {rating}/10, Time {time_str} {cache_status}")

            return result

        except Exception as e:
            import traceback

            error_msg = str(e)
            print(f"Error on problem {problem_index + 1}: {error_msg}")
            return {
                "problem_index": problem_index,
                "problem_number": problem.get("problem_number", "N/A"),
                "category": problem.get("category", "N/A"),
                "round": problem.get("round", "N/A"),
                "difficulty_rating": None,
                "solve_time_minutes": None,
                "error": error_msg,
                "problem": problem["problem"],
                "answer": problem.get("answer"),
                "cached": False,
            }


async def run_rating(
    input_file,
    output_file=None,
    max_problems=None,
    concurrency=20,
    model="claude-opus-4-5-20251101",
):
    """
    Rate all problems and save results.
    """
    problems = load_problems(input_file)

    if max_problems:
        problems = problems[:max_problems]

    print(f"Rating {len(problems)} problems with concurrency={concurrency}...")

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [rate_problem(problem, semaphore, i, model) for i, problem in enumerate(problems)]

    results = await asyncio.gather(*tasks)

    # Save any remaining cached responses
    await response_cache.save_cache(force=True)

    # Sort results by problem_index
    results = sorted(results, key=lambda x: x["problem_index"])

    # Calculate statistics
    ratings = [r["difficulty_rating"] for r in results if r["difficulty_rating"]]
    solve_times = [r["solve_time_minutes"] for r in results if r["solve_time_minutes"]]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    avg_time = sum(solve_times) / len(solve_times) if solve_times else 0
    cached_count = sum(1 for r in results if r.get("cached", False))

    print(f"\n{'='*60}")
    print(f"RATING COMPLETE")
    print(f"{'='*60}")
    print(f"Total problems: {len(results)}")
    print(f"Successfully rated: {len(ratings)}")
    print(f"Average difficulty: {avg_rating:.2f}/10")
    print(f"Average solve time: {avg_time:.1f} minutes")
    print(f"Cache hits: {cached_count}/{len(results)}")

    # Difficulty distribution
    print(f"\nDifficulty distribution:")
    for i in range(1, 11):
        count = sum(1 for r in ratings if r == i)
        bar = "█" * round(count * 0.25)
        print(f"  {i:2d}: {bar} ({count})")

    # Solve time distribution (buckets)
    print(f"\nSolve time distribution:")
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
    for (low, high), label in zip(time_buckets, bucket_labels):
        count = sum(1 for t in solve_times if low <= t < high)
        bar = "█" * round(count * 0.25)
        print(f"  {label:>5} min: {bar} ({count})")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "total": len(results),
                        "rated": len(ratings),
                        "average_difficulty": avg_rating,
                        "average_solve_time_minutes": avg_time,
                        "cached": cached_count,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_file}")

    # Plot histograms
    plot_histograms(ratings, solve_times, time_buckets)

    return results


def plot_histograms(ratings, solve_times, time_buckets):
    """Plot difficulty and solve time histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Difficulty histogram
    ax1 = axes[0]
    difficulty_counts = [sum(1 for r in ratings if r == i) for i in range(1, 11)]
    ax1.bar(range(1, 11), difficulty_counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Difficulty Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Difficulty Distribution')
    ax1.set_xticks(range(1, 11))

    # Solve time histogram
    ax2 = axes[1]
    bucket_counts = []
    bucket_labels = []
    for low, high in time_buckets:
        count = sum(1 for t in solve_times if low <= t < high)
        bucket_counts.append(count)
        if high == float("inf"):
            bucket_labels.append(f"{int(low)}+")
        else:
            bucket_labels.append(f"{low}-{high}")

    ax2.bar(range(len(bucket_counts)), bucket_counts, color='coral', edgecolor='black')
    ax2.set_xlabel('Solve Time (minutes)')
    ax2.set_ylabel('Count')
    ax2.set_title('Solve Time Distribution')
    ax2.set_xticks(range(len(bucket_labels)))
    ax2.set_xticklabels(bucket_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('difficulty_histograms.png', dpi=150)
    plt.show()
    print(f"\nHistograms saved to: difficulty_histograms.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate math problem difficulty")
    parser.add_argument(
        "--num-problems",
        "-n",
        type=int,
        default=None,
        help="Number of problems to rate (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=20,
        help="Number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="problems_with_answers.jsonl",
        help="Input JSONL file (default: problems_with_answers.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="eval_results/difficulty_ratings.json",
        help="Output JSON file (default: eval_results/difficulty_ratings.json)",
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max problems: {args.num_problems if args.num_problems else 'all'}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")

    asyncio.run(
        run_rating(
            args.input,
            args.output,
            args.num_problems,
            args.concurrency,
        )
    )
