#!/usr/bin/env python3
"""
Evaluation script for math problems using Claude with extended thinking.
"""

import json
import re
import os
import asyncio
from anthropic import AsyncAnthropic
from response_cache import ResponseCache

if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key_rr")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

# Initialize Anthropic client
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize cache (signal handlers registered automatically by response_cache module)
response_cache = ResponseCache("caches/reasoning_cache.json")


def load_problems(filepath):
    """Load problems from JSONL file."""
    problems = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def extract_boxed_answer(text):
    """
    Extract the numerical answer from boxed notation.
    Supports formats like \\boxed{123}, \\boxed{123.45}, etc.
    """
    # Look for \boxed{...} pattern
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)

    if matches:
        # Take the last boxed answer (final answer)
        answer_text = matches[-1].strip()
        # Extract number from the answer text
        number_match = re.search(r"-?\d+\.?\d*", answer_text)
        if number_match:
            num_str = number_match.group()
            # Try to parse as int first, then float
            try:
                return int(num_str)
            except ValueError:
                try:
                    return float(num_str)
                except ValueError:
                    return None
    return None


async def evaluate_problem(problem, semaphore, problem_index, model="claude-opus-4-5-20251101"):
    """
    Send a problem to Claude with extended thinking and extract the answer.

    Args:
        problem: dict with 'problem' and 'answer' keys
        semaphore: asyncio.Semaphore to limit concurrency
        problem_index: index of the problem for logging
        model: Claude model to use

    Returns:
        dict with evaluation results
    """
    prompt = f"""Solve the following math problem and provide your final answer in \\boxed{{answer}} notation.

Problem: {problem['problem']}"""

    # Define API call parameters
    max_tokens = 16_000
    thinking_config = {"type": "enabled", "budget_tokens": 12_000}
    messages = [{"role": "user", "content": prompt}]

    # Check cache first (outside semaphore to avoid blocking)
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
                # Use cached response
                print(
                    f"[CACHED] Problem {problem_index + 1}: Category {problem.get('category')}, Problem {problem.get('problem_number')}"
                )
                thinking_text = cached_response.get("thinking", "")
                response_text = cached_response.get("response", "")
            else:
                # Make API call
                print(
                    f"Starting problem {problem_index + 1}: Category {problem.get('category')}, Problem {problem.get('problem_number')}"
                )

                # Add timeout handling (15 minutes)
                try:
                    response = await asyncio.wait_for(
                        client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            thinking=thinking_config,
                            messages=messages,
                            timeout=600.0,  # 10 minute timeout per request
                        ),
                        timeout=900.0,  # 15 minute overall timeout
                    )

                    # Extract thinking and response text
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
                    print(f"TIMEOUT on problem {problem_index + 1} after 15 minutes")
                    raise Exception("API call timed out after 15 minutes")

            # Extract boxed answer
            predicted_answer = extract_boxed_answer(response_text)
            correct_answer = problem["answer"]

            # Check if correct
            is_correct = predicted_answer == correct_answer

            result = {
                "problem_index": problem_index,
                "problem_number": problem.get("problem_number", "N/A"),
                "category": problem.get("category", "N/A"),
                "round": problem.get("round", "N/A"),
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "thinking": thinking_text,
                "response": response_text,
                "problem": problem["problem"],
                "cached": cached_response is not None,
            }

            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            cache_status = "[CACHED]" if cached_response else ""
            print(
                f"Completed problem {problem_index + 1}: {status} ({predicted_answer} vs {correct_answer}) {cache_status}"
            )

            return result

        except Exception as e:
            import traceback

            error_msg = str(e)
            full_traceback = traceback.format_exc()
            print(f"Error on problem {problem_index + 1}:")
            print(f"  Error message: {error_msg}")
            print(f"  Full error:\n{full_traceback}")
            return {
                "problem_index": problem_index,
                "problem_number": problem.get("problem_number", "N/A"),
                "category": problem.get("category", "N/A"),
                "round": problem.get("round", "N/A"),
                "correct_answer": problem["answer"],
                "predicted_answer": None,
                "is_correct": False,
                "error": error_msg,
                "error_traceback": full_traceback,
                "problem": problem["problem"],
                "cached": False,
            }


async def run_evaluation(
    input_file,
    output_file=None,
    max_problems=None,
    concurrency=20,
    model="claude-opus-4-5-20251101",
):
    """
    Run evaluation on all problems and save results.

    Args:
        input_file: path to JSONL file with problems
        output_file: path to save results (optional)
        max_problems: maximum number of problems to evaluate (optional)
        concurrency: maximum number of concurrent API calls (default: 20)
        model: Claude model to use (default: claude-opus-4-5-20251101)
    """
    # Load problems
    problems = load_problems(input_file)

    if max_problems:
        problems = problems[:max_problems]

    print(f"Evaluating {len(problems)} problems with concurrency={concurrency}...")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all problems
    tasks = [evaluate_problem(problem, semaphore, i, model) for i, problem in enumerate(problems)]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Save any remaining cached responses
    await response_cache.save_cache(force=True)

    # Sort results by problem_index to maintain order
    results = sorted(results, key=lambda x: x["problem_index"])

    # Calculate statistics
    correct_count = sum(1 for r in results if r["is_correct"])
    cached_count = sum(1 for r in results if r.get("cached", False))
    accuracy = correct_count / len(problems) if problems else 0
    cache_hit_rate = cached_count / len(problems) if problems else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total problems: {len(problems)}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(problems) - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Cache hits: {cached_count}/{len(problems)} ({cache_hit_rate:.1%})")

    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "total": len(problems),
                        "correct": correct_count,
                        "incorrect": len(problems) - correct_count,
                        "accuracy": accuracy,
                        "cached": cached_count,
                        "cache_hit_rate": cache_hit_rate,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_file}")

    return results


def parse_model_name(model_shorthand):
    """Convert model shorthand to full model ID."""
    model_map = {
        "opus-4-5": "claude-opus-4-5-20251101",
        "sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus-4-1": "claude-opus-4-1-20250805",
        "opus-4": "claude-opus-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        "haiku-3": "claude-3-haiku-20240307",
    }
    return model_map.get(model_shorthand, model_shorthand)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Claude on math problems with extended thinking")
    parser.add_argument(
        "--num-problems",
        "-n",
        type=int,
        default=None,
        help="Number of problems to evaluate (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=20,
        help="Number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="opus-4-5",
        help="Model to use. Supports shorthands: opus-4-5, sonnet-4-5, opus-4, sonnet-4, or full model ID (default: opus-4-5)",
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
        default="eval_results/eval_results.json",
        help="Output JSON file (default: eval_results/eval_results.json)",
    )

    args = parser.parse_args()

    # Parse model name
    model = parse_model_name(args.model)

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max problems: {args.num_problems if args.num_problems else 'all'}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")

    # Run evaluation
    asyncio.run(run_evaluation(args.input, args.output, args.num_problems, args.concurrency, model))
