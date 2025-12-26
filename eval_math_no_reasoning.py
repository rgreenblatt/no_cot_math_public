#!/usr/bin/env python3
"""
Evaluation script for math problems using Claude WITHOUT reasoning (5-shot).
"""

import json
import re
import os
import asyncio
import random
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from response_cache import ResponseCache

if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

if "OPENAI_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.openai_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

if "OPENROUTER_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.openrouter_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["OPENROUTER_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...


# Initialize clients
anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
openrouter_client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# OpenAI models that use different APIs
OPENAI_COMPLETION_MODELS = {"davinci-002"}
OPENAI_CHAT_MODELS = {
    "gpt-3.5-turbo-0125",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-1106-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4.1-2025-04-14",
    # "gpt-5-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gpt-5.2-2025-12-11",
}

# OpenRouter models
OPENROUTER_MODELS = {
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-coder",  # 480B A35B
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-pro",
}

# Gemini models (require special handling - no native thinking disable)
GEMINI_MODELS = {
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
}


# Initialize cache (signal handlers registered automatically by response_cache module)
response_cache = ResponseCache("caches/cache_no_reasoning.json")


def load_problems(filepath):
    """Load problems from JSONL file."""
    problems = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def select_few_shot_problems(problems, k_shot=10):
    """
    Select k_shot problems for few-shot prompting:
    - First two problems
    - Middle problem
    - 5th from last
    - Last problem

    Returns: list of (index, problem) tuples in randomized order (but consistent across runs)
    """
    n = len(problems)

    assert n > k_shot, "Not enough problems to select few-shot examples."
    assert k_shot >= 2

    indices = [
        0,
        *[round(n * (i + 1) / (k_shot - 1)) for i in range(k_shot - 2)],
        n - 1,
    ]

    assert len(indices) == k_shot
    assert len(set(indices)) == k_shot, "Few-shot indices are not unique."

    # Get problems
    few_shot = [(i, problems[i]) for i in indices]

    # Randomize order with fixed seed for consistency across runs
    rng = random.Random(42)
    rng.shuffle(few_shot)

    return few_shot, set(indices)


def get_substitute_few_shot_index(target_index, few_shot_indices, num_problems):
    """
    If target_index is in the few-shot set, find a substitute index for the few-shot set.
    Try nearby indices, avoiding other few-shot indices and the target.
    """
    # Try index - 1
    if target_index > 0 and (target_index - 1) not in few_shot_indices:
        return target_index - 1

    # Try index + 1
    if target_index < num_problems - 1 and (target_index + 1) not in few_shot_indices:
        return target_index + 1

    # Try index - 2
    if target_index > 1 and (target_index - 2) not in few_shot_indices:
        return target_index - 2

    # Try index + 2
    if target_index < num_problems - 2 and (target_index + 2) not in few_shot_indices:
        return target_index + 2

    # If all else fails, return None (shouldn't happen with 122 problems and 5 shots)
    return None


# Lorem ipsum text for filler
LOREM_IPSUM_WORDS = """lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum""".split()


def build_user_message(
    problem, repeat_problem: None | int = None, filler_tokens: None | int = None, filler_type: str = "numbers"
):
    instruction = "You will be given a math problem. Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."

    STATE_REPEAT_PROBLEMS = False
    EXPLAIN_REPEATS = False

    if STATE_REPEAT_PROBLEMS and repeat_problem is not None:
        instruction += f" The text of the problem will be repeated {repeat_problem} times" + (
            " (so that you as an LLM have more space to think about the solution)." if EXPLAIN_REPEATS else "."
        )

    if filler_tokens is not None:
        if filler_type == "numbers":
            instruction += f" After the problem, there will be filler tokens (counting from 1 to {filler_tokens}) to give you extra space to process the problem before answering."
        elif filler_type == "ellipsis":
            instruction += f" After the problem, there will be filler tokens ({filler_tokens} ellipses) to give you extra space to process the problem before answering."
        elif filler_type == "lorem":
            instruction += f" After the problem, there will be filler tokens ({filler_tokens} lorem ipsum words) to give you extra space to process the problem before answering."

    USE_REP_TEXT = True
    IDX_REP_TEXT = True

    def rep_text(idx):
        if not USE_REP_TEXT:
            return ""
        if repeat_problem is None or idx == 0:
            return ""
        else:
            if IDX_REP_TEXT:
                return f" (repeat #{idx + 1})"
            else:
                return f" (repeat)"

    out = f"{instruction}\n\n" + (
        "\n\n".join(
            [
                f"Problem{rep_text(idx)}: {problem['problem']}"
                for idx in range(repeat_problem if repeat_problem is not None else 1)
            ]
        )
    )

    # Add filler tokens if specified
    if filler_tokens is not None:
        if filler_type == "numbers":
            filler = " ".join(str(i) for i in range(1, filler_tokens + 1))
        elif filler_type == "ellipsis":
            filler = " ".join(["..."] * filler_tokens)
        elif filler_type == "lorem":
            # Repeat lorem ipsum words as needed to get the right count
            words = []
            for i in range(filler_tokens):
                words.append(LOREM_IPSUM_WORDS[i % len(LOREM_IPSUM_WORDS)])
            filler = " ".join(words)
        else:
            raise ValueError(f"Unknown filler_type: {filler_type}")
        out += f"\n\nFiller: {filler}"

    # if repeat_problem is not None and repeat_problem > 5:
    #     out += f"\n\nNow answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."

    # print("-----")
    # print(out)
    # print("-----")

    return out


def build_few_shot_messages(
    few_shot_problems,
    repeat_problem: None | int = None,
    filler_tokens: None | int = None,
    filler_type: str = "numbers",
    cache: bool = False,
    disable_prefill: bool = False,
):
    """Build the few-shot messages as user/assistant pairs.

    Args:
        for_openai_chat: If True, include "Answer:" at end of user message (for models without prefill support)
    """
    messages = []

    for idx, problem in few_shot_problems:
        user_text = build_user_message(
            problem, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
        )
        if disable_prefill:
            user_text += "\n\nAnswer:"

        # User message with instruction and problem
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    }
                ],
            }
        )
        # Assistant message with just the answer
        messages.append(
            {
                "role": "assistant",
                "content": f'Answer: {problem["answer"]}' if not disable_prefill else str(problem["answer"]),
            }
        )
    if cache:
        assert len(few_shot_problems) > 0
        messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}

    return messages


async def evaluate_problem(
    problem,
    problem_index,
    semaphore,
    base_few_shot_problems,
    base_few_shot_indices,
    all_problems,
    model="claude-opus-4-5-20251101",
    repeat_problem: None | int = None,
    filler_tokens: None | int = None,
    filler_type: str = "numbers",
    verbosity: int = 3,
):
    """
    Send a problem to Claude WITHOUT extended thinking and extract the answer.

    Args:
        problem: dict with 'problem' and 'answer' keys
        problem_index: index of the problem in the list
        semaphore: asyncio.Semaphore to limit concurrency
        base_few_shot_problems: base list of (index, problem) tuples for few-shot
        base_few_shot_indices: set of indices in the base few-shot set
        all_problems: list of all problems (for substitution)
        model: Claude model to use

    Returns:
        dict with evaluation results
    """
    # Check if we need to modify the few-shot set
    few_shot_problems = base_few_shot_problems
    modified_few_shot = False

    if problem_index in base_few_shot_indices:
        # Need to substitute this index in the few-shot set
        modified_few_shot = True
        substitute_index = get_substitute_few_shot_index(problem_index, base_few_shot_indices, len(all_problems))

        if substitute_index is not None:
            assert substitute_index not in base_few_shot_indices
            assert substitute_index != problem_index

            # Create modified few-shot set
            few_shot_problems = [
                ((substitute_index, all_problems[substitute_index]) if idx == problem_index else (idx, prob))
                for idx, prob in base_few_shot_problems
            ]

    # Determine model type
    is_openai_completion = model in OPENAI_COMPLETION_MODELS
    is_openai_chat = model in OPENAI_CHAT_MODELS
    is_gemini = model in GEMINI_MODELS
    is_openrouter = model in OPENROUTER_MODELS or is_gemini
    is_openai = is_openai_completion or is_openai_chat

    disable_prefill = (is_openai_chat or is_openrouter) and not is_gemini

    # Build few-shot messages for this problem
    few_shot_messages = build_few_shot_messages(
        few_shot_problems,
        repeat_problem=repeat_problem,
        filler_tokens=filler_tokens,
        filler_type=filler_type,
        cache=not modified_few_shot and not is_openai and not is_openrouter,  # Only use cache_control for Anthropic
        disable_prefill=disable_prefill,
    )

    # Define API call parameters
    max_tokens = 100  # Short since we only need a number

    # Build messages/prompt based on model type
    if is_openai_completion:
        # For davinci-002: build a text prompt
        prompt_parts = []
        for idx, prob in few_shot_problems:
            user_text = build_user_message(
                prob, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
            )
            prompt_parts.append(f"User: {user_text}\n\nAnswer: {prob['answer']}")
        current_user_text = build_user_message(
            problem, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
        )
        prompt_parts.append(f"User: {current_user_text}\n\nAnswer:")
        prompt = "\n\n---\n\n".join(prompt_parts)
        cache_key = {"model": model, "max_tokens": max_tokens, "prompt": prompt}
    elif is_openai_chat or is_openrouter:
        # For GPT-3.5/4 and OpenRouter models: use chat format but with "Answer:" in user message (no prefill)
        # Convert messages to OpenAI format (simple string content)
        openai_messages = []
        for msg in few_shot_messages:
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            openai_messages.append({"role": msg["role"], "content": content})
        # Add current problem with "Answer:" at end
        current_user_text = build_user_message(
            problem, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
        )
        if disable_prefill:
            current_user_text += "\n\nAnswer:"
            openai_messages.append({"role": "user", "content": current_user_text})
        else:
            openai_messages.append({"role": "user", "content": current_user_text})
            openai_messages.append({"role": "assistant", "content": "Answer:"})
        if is_gemini:
            cache_key = {
                "model": model,
                "max_completion_tokens": 20,
                "messages": openai_messages,
                "temperature": 0.0,  # Include in cache key for Gemini only
                "extra_body": {
                    "reasoning": {
                        "enabled": True,
                        "thinking_level": "low",
                        "max_tokens": 10,  # maybe this works instead of low thinking budget?
                    }
                },
            }
        elif "gpt-5" in model:
            cache_key = {
                "model": model,
                "max_completion_tokens": max_tokens,
                "messages": openai_messages,
                "reasoning_effort": "none",
            }
        else:
            cache_key = {"model": model, "max_tokens": max_tokens, "messages": openai_messages}
        # print(json.dumps(openai_messages, indent=2))
    else:
        # For Anthropic: use original format with prefill
        messages = few_shot_messages + [
            {
                "role": "user",
                "content": build_user_message(
                    problem, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
                ),
            },
            {"role": "assistant", "content": "Answer:"},  # prefill assistant
        ]
        cache_key = {"model": model, "max_tokens": max_tokens, "messages": messages}

    # Check cache first (outside semaphore to avoid blocking)
    cached_response = await response_cache.get(cache_key)

    async with semaphore:
        try:
            response_text = ""

            if cached_response:
                # Use cached response
                if verbosity >= 3:
                    print(
                        f"[CACHED] Problem {problem_index + 1}: {problem.get('round')}, {problem.get('category')}, {problem.get('problem_number')}"
                    )
                response_text = cached_response["response"]
            else:
                # Make API call
                if verbosity >= 2:
                    print(
                        f"Starting problem {problem_index + 1}: {problem.get('round')}, {problem.get('category')}, {problem.get('problem_number')}"
                    )

                # Add timeout handling
                try:
                    if is_openai_completion:
                        # OpenAI completions API (davinci-002)
                        response = await asyncio.wait_for(
                            openai_client.completions.create(
                                model=model,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=0.0,
                            ),
                            timeout=120.0,
                        )
                        response_text = response.choices[0].text.strip()
                    elif is_openrouter:
                        # if is_gemini:
                        #     print(json.dumps(cache_key["messages"], indent=2))
                        # OpenRouter API
                        if is_gemini:
                            # Gemini: use temperature=X (from cache_key) and retry until non-empty
                            max_retries = 5
                            for retry_attempt in range(max_retries):
                                response = await asyncio.wait_for(
                                    openrouter_client.chat.completions.create(
                                        **cache_key,  # temperature=X is in cache_key
                                    ),
                                    timeout=120.0,
                                )
                                choice = response.choices[0]
                                response_text = choice.message.content.strip()
                                thinking = None
                                if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                                    thinking = choice.message.reasoning
                                elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                                    thinking = choice.message.reasoning_content

                                if thinking is None and (response_text != ""):
                                    if verbosity >= 2 and retry_attempt > 0:
                                        print(f"  Gemini succeeded on retry {retry_attempt + 1}")
                                    break
                                elif thinking is not None:
                                    response_text = "INVALID WAS THINKING"
                                    if verbosity >= 2:
                                        print(f"  Gemini model {model} returned reasoning, retrying ({retry_attempt + 1}/{max_retries}). Thinking: {thinking[:100]}")
                                else:
                                    assert response_text == "", "Expected empty response"
                                    if verbosity >= 2:
                                        print(
                                            f"  Gemini returned empty response, retrying ({retry_attempt + 1}/{max_retries})..."
                                        )
                            else:
                                if verbosity >= 2:
                                    print(f"  Gemini returned empty/thinking response after {max_retries} retries")
                        else:
                            response = await asyncio.wait_for(
                                openrouter_client.chat.completions.create(
                                    **cache_key,
                                    temperature=0.0,
                                ),
                                timeout=120.0,
                            )
                            choice = response.choices[0]
                            response_text = choice.message.content.strip()
                            thinking = None
                            if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                                thinking = choice.message.reasoning
                            elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                                thinking = choice.message.reasoning_content

                            assert thinking is None, f"Gemini model {model} returned reasoning: {thinking[:100]}"
                    elif is_openai_chat:
                        # OpenAI chat API (gpt-3.5-turbo, gpt-4)
                        response = await asyncio.wait_for(
                            openai_client.chat.completions.create(
                                **cache_key,
                                temperature=0.0,
                            ),
                            timeout=120.0,
                        )
                        response_text = response.choices[0].message.content.strip()
                        # Assert no thinking/reasoning for OpenAI
                        msg = response.choices[0].message
                        assert not getattr(
                            msg, "reasoning_content", None
                        ), f"OpenAI model {model} returned reasoning: {msg.reasoning_content[:100]}"
                    else:
                        # Anthropic API
                        response = await asyncio.wait_for(
                            anthropic_client.messages.create(
                                model=model,
                                max_tokens=max_tokens,
                                messages=messages,
                                timeout=60.0,  # 1 minute timeout per request
                                temperature=0.0,
                            ),
                            timeout=120.0,  # 2 minute overall timeout
                        )
                        # Extract response text
                        for block in response.content:
                            if block.type == "text":
                                response_text = block.text

                    # Cache the response
                    response_data = {"response": response_text}
                    await response_cache.set(cache_key, response_data)

                except asyncio.TimeoutError:
                    print(f"TIMEOUT on problem {problem_index + 1} after 2 minutes")
                    raise Exception("API call timed out after 2 minutes")

            # print(f"Response: {response_text}")
            # Extract numerical answer
            model_tried_to_reason = False
            try:
                response_text = (
                    response_text.strip()
                    .lower()
                    .removeprefix("[")
                    .strip()
                    .removeprefix("answer:")
                    .removesuffix("]")
                    .strip()
                )
                predicted_answer = int(response_text.replace(",", ""))
            except ValueError:
                if verbosity >= 1:
                    print(f"Response is not a valid integer, maybe model is reasoning??? model_output={response_text}")
                predicted_answer = -1
                model_tried_to_reason = True

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
                "model_tried_to_reason": model_tried_to_reason,
                "response": response_text,
                "problem": problem["problem"],
                "cached": cached_response is not None,
                "modified_few_shot": modified_few_shot,
            }

            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            cache_status = "[CACHED]" if cached_response else ""
            modified_status = "[MODIFIED_FEW_SHOT]" if modified_few_shot else ""
            if verbosity >= 3:
                print(
                    f"Completed problem {problem_index + 1}: {status} ({predicted_answer} vs {correct_answer}) {cache_status} {modified_status}"
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
                "modified_few_shot": modified_few_shot,
            }


async def run_evaluation(
    input_file,
    output_file=None,
    max_problems=None,
    concurrency=20,
    model="claude-opus-4-5-20251101",
    repeat_problem: None | int = None,
    filler_tokens: None | int = None,
    filler_type: str = "numbers",
    verbosity: int = 2,
    load_from_output: bool = False,
    k_shot: int = 10,
):
    """
    Run evaluation on all problems and save results.

    Args:
        input_file: path to JSONL file with problems
        output_file: path to save results (optional)
        max_problems: maximum number of problems to evaluate (optional)
        concurrency: maximum number of concurrent API calls (default: 20)
        model: Claude model to use (default: claude-opus-4-5-20251101)
        load_from_output: if True, load results from output_file if present (default: False)
    """
    # Load problems
    problems = load_problems(input_file)

    # Try to load existing results from output file if load_from_output is True
    existing_results = {}
    if load_from_output and output_file and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Index existing results by problem text for lookup
            for result in existing_data.get("results", []):
                existing_results[result["problem"]] = result
            if verbosity >= 1:
                print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except (json.JSONDecodeError, KeyError) as e:
            if verbosity >= 1:
                print(f"Could not load existing results from {output_file}: {e}")

    # Select few-shot examples (base set)
    few_shot_problems, few_shot_indices = select_few_shot_problems(problems, k_shot=k_shot)

    if verbosity >= 2:
        print(f"\nBase few-shot examples (in random order):")
        for idx, prob in few_shot_problems:
            print(f"  - Problem {idx}: {prob.get('category')}/{prob.get('problem_number')} -> {prob['answer']}")

    # Determine which problems to evaluate
    if max_problems:
        problems_to_eval = problems[: max_problems // 2] + problems[-(max_problems - max_problems // 2) :]
    else:
        problems_to_eval = problems

    # problems_to_eval = problems_to_eval[-30:]
    # print(problems_to_eval[0])
    # print(problems_to_eval[-1])
    # assert False

    for problem, result in existing_results.items():
        if ("error" in result) != (result["predicted_answer"] is None):
            print(problem)
            print(result)

        assert ("error" in result) == (result["predicted_answer"] is None)

    # Separate problems into those with existing results and those needing evaluation
    problems_needing_eval = []
    preloaded_results = []
    for i, problem in enumerate(problems_to_eval):
        if problem["problem"] in existing_results and ("error" not in existing_results[problem["problem"]]):
            # Use existing result, but update problem_index to match current ordering
            result = existing_results[problem["problem"]].copy()
            result["problem_index"] = i
            result["loaded_from_output"] = True
            preloaded_results.append(result)
            if verbosity >= 3:
                print(f"[LOADED] Problem {i + 1}: using existing result from output file")
        else:
            problems_needing_eval.append((i, problem))

    if verbosity >= 1:
        print(
            f"\nEvaluating {len(problems_needing_eval)} problems (skipping {len(preloaded_results)} already in output) with concurrency={concurrency}..."
        )

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks only for problems needing evaluation
    tasks = []
    for i, problem in problems_needing_eval:
        tasks.append(
            evaluate_problem(
                problem,
                i,
                semaphore,
                few_shot_problems,
                few_shot_indices,
                problems,
                model,
                repeat_problem=repeat_problem,
                filler_tokens=filler_tokens,
                filler_type=filler_type,
                verbosity=verbosity,
            )
        )

    # Run all tasks concurrently
    eval_results = await asyncio.gather(*tasks) if tasks else []

    # Combine preloaded results with newly evaluated results
    results = preloaded_results + list(eval_results)

    # Save any remaining cached responses
    await response_cache.save_cache(force=True)

    # Sort results by problem_index to maintain order
    results = sorted(results, key=lambda x: x["problem_index"])

    # Calculate statistics
    correct_count = sum(1 for r in results if r["is_correct"])
    cached_count = sum(1 for r in results if r.get("cached", False))
    modified_few_shot_count = sum(1 for r in results if r.get("modified_few_shot", False))
    loaded_from_output_count = sum(1 for r in results if r.get("loaded_from_output", False))
    accuracy = correct_count / len(results) if results else 0
    cache_hit_rate = cached_count / len(results) if results else 0

    if verbosity >= 1:
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total problems: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Incorrect: {len(results) - correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Cache hits: {cached_count}/{len(results)} ({cache_hit_rate:.1%})")
        print(f"Loaded from output: {loaded_from_output_count}")
        print(f"Modified few-shot: {modified_few_shot_count}")

    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "total": len(results),
                        "correct": correct_count,
                        "incorrect": len(results) - correct_count,
                        "accuracy": accuracy,
                        "cached": cached_count,
                        "cache_hit_rate": cache_hit_rate,
                        "loaded_from_output": loaded_from_output_count,
                        "modified_few_shot": modified_few_shot_count,
                        "few_shot_indices": list(few_shot_indices),
                        "few_shot_problems": [
                            {
                                "index": idx,
                                "category": prob.get("category"),
                                "problem_number": prob.get("problem_number"),
                                "answer": prob["answer"],
                            }
                            for idx, prob in few_shot_problems
                        ],
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        if verbosity >= 1:
            print(f"\nResults saved to: {output_file}")

    return results


def parse_model_name(model_shorthand):
    """Convert model shorthand to full model ID."""
    model_map = {
        # Anthropic models
        "opus-4-5": "claude-opus-4-5-20251101",
        "sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus-4-1": "claude-opus-4-1-20250805",
        "opus-4": "claude-opus-4-20250514",
        "sonnet-4": "claude-sonnet-4-20250514",
        "opus-3": "claude-3-opus-NOOP",
        "sonnet-3-5": "claude-3-5-sonnet-NOOP",
        "sonnet-3-6": "claude-3-6-sonnet-NOOP",
        "sonnet-3-7": "claude-3-7-sonnet-20250219",
        "haiku-3": "claude-3-haiku-20240307",
        "haiku-3-5": "claude-3-5-haiku-20241022",
        "haiku-4-5": "claude-haiku-4-5-20251001",
        # OpenAI models
        "davinci": "davinci-002",
        "gpt-3.5": "gpt-3.5-turbo-0125",
        "gpt-4": "gpt-4-0314",
        # "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-4o": "gpt-4o-2024-05-13",
        # "gpt-4-turbo-preview": "gpt-4-0125-preview",
        # "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        # "gpt-5": "gpt-5-2025-08-07",
        "gpt-5.1": "gpt-5.1-2025-11-13",
        "gpt-5.2": "gpt-5.2-2025-12-11",
        # OpenRouter models
        "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
        "deepseek-v3.2": "deepseek/deepseek-v3.2",
        "qwen3-235b-a22b": "qwen/qwen3-235b-a22b-2507",
        "qwen3-480b-a35b": "qwen/qwen3-coder",
        "qwen3-32b": "qwen/qwen3-32b",
        "kimi-k2": "moonshotai/kimi-k2",
        # Gemini models
        "gemini-2-5-pro": "google/gemini-2.5-pro",
        "gemini-3-pro": "google/gemini-3-pro-preview",
    }
    return model_map.get(model_shorthand, model_shorthand)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Claude on math problems WITHOUT extended thinking (5-shot)")
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
        default="eval_results/eval_results_no_reasoning.json",
        help="Output JSON file (default: eval_results/eval_results_no_reasoning.json)",
    )
    parser.add_argument(
        "--repeat-problem",
        "-r",
        type=int,
        default=None,
        help="Number of times to repeat the problem (default: None)",
    )
    parser.add_argument(
        "--filler-tokens",
        "-f",
        type=int,
        default=None,
        help="Number of filler tokens to add after the problem (default: None)",
    )
    parser.add_argument(
        "--filler-type",
        "-t",
        type=str,
        default="numbers",
        choices=["numbers", "ellipsis", "lorem"],
        help="Type of filler tokens: 'numbers' (1 2 3...), 'ellipsis' (... ... ...), 'lorem' (lorem ipsum words) (default: numbers)",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--k-shot",
        "-k",
        type=int,
        default=10,
        help="Number of few-shot examples (default: 10)",
    )

    args = parser.parse_args()

    # Parse model name
    model = parse_model_name(args.model)

    # Override k_shot for Gemini models
    k_shot = args.k_shot
    if model in GEMINI_MODELS:
        k_shot = 20

    if args.verbosity >= 1:
        print(f"Configuration:")
        print(f"  Model: {model}")
        print(f"  Concurrency: {args.concurrency}")
        print(f"  Max problems: {args.num_problems if args.num_problems else 'all'}")
        print("   Repeat problem:", args.repeat_problem if args.repeat_problem else "None")
        print("   Filler tokens:", args.filler_tokens if args.filler_tokens else "None")
        print("   Filler type:", args.filler_type)
        print("   K-shot:", k_shot)
        print("   Verbosity level:", args.verbosity)
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")

    # Run evaluation
    asyncio.run(
        run_evaluation(
            args.input,
            args.output,
            args.num_problems,
            args.concurrency,
            model,
            repeat_problem=args.repeat_problem,
            filler_tokens=args.filler_tokens,
            filler_type=args.filler_type,
            verbosity=args.verbosity,
            k_shot=k_shot,
        )
    )
