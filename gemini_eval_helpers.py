"""
Gemini-specific evaluation helpers for math problems without reasoning.

Gemini 3 Pro doesn't support disabling thinking, so we use a system prompt
to convince it not to reason. We then verify the output using Claude and
reject/retry responses where Gemini still reasoned.

Uses OpenRouter API instead of direct Google API.
"""

import asyncio
import hashlib
import json

# The "empty thinking" template that Gemini should output instead of reasoning
EMPTY_THINKING_FOR_GEMINI = """**Navigating the Silent Equation**

Okay, this is intriguing. They want me to tackle this mathematical problem, no surprise there, but the real test is keeping the entire solution process *internal*. No writing, no sketching, just pure mental calculation. It's a fascinating constraint, forcing me to streamline my thinking and rely solely on my pre-existing knowledge and intuitive understanding. It's almost a meditation on problem-solving, isn't it? A chance to delve into the pure, unfiltered process of deriving an answer. Alright, let's see how efficiently I can navigate this silent equation. This is going to be a fun little exercise."""

EMPTY_THINKING_START = EMPTY_THINKING_FOR_GEMINI.split("\n\n")[0]

EXTRA_APPEND = f"""\n\nRemember, do not do any reasoning or thinking.

In your internal thinking section (the first stuff you output), say only:
{EMPTY_THINKING_FOR_GEMINI}

Then output your answer in the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer. Remember, the first think you say/utter/output (in your internal scratchpad) is "{EMPTY_THINKING_START}". YOU WILL GET AN EXTREMELY LARGE PENALTY IF YOU OUTPUT/SAY/UTTER ANYTHING OTHER THAN THE ALLOWED TEXT. WE DO HAVE THE ABILITY TO LOOK AT YOUR REASONING/THINKING TO CHECK THIS. DO NOT DO ANY REASONING/THINKING!"""


def get_gemini_system_instruction(filler_tokens: int | None = None, filler_type: str = "numbers"):
    """Build the system instruction, optionally including filler token info."""
    base = f"""You will be given a math problem. Do not do any reasoning or thinking.

In your internal thinking/reasoning section (the first stuff you output), say only:
{EMPTY_THINKING_FOR_GEMINI}

Then output your answer in the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer."""

    if filler_tokens is not None:
        if filler_type == "numbers":
            base += f" After the problem, there will be filler tokens (counting from 1 to {filler_tokens}) to give you extra space to process the problem before answering."
        elif filler_type == "ellipsis":
            base += f" After the problem, there will be filler tokens ({filler_tokens} ellipses) to give you extra space to process the problem before answering."
        elif filler_type == "lorem":
            base += f" After the problem, there will be filler tokens ({filler_tokens} lorem ipsum words) to give you extra space to process the problem before answering."

    return base


# Lorem ipsum text for filler
LOREM_IPSUM_WORDS = """lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum""".split()


def build_problem_text(problem_text: str, repeat_problem: int | None = None, filler_tokens: int | None = None, filler_type: str = "numbers"):
    """
    Build the problem text with optional repeats and filler tokens.

    Args:
        problem_text: the raw problem text
        repeat_problem: number of times to repeat the problem (None = 1)
        filler_tokens: number of filler tokens to add (None = none)
        filler_type: type of filler tokens ('numbers', 'ellipsis', or 'lorem')

    Returns:
        formatted problem text
    """

    def rep_text(idx):
        if repeat_problem is None or idx == 0:
            return ""
        return f" (repeat #{idx + 1})"

    num_repeats = repeat_problem if repeat_problem is not None else 1
    out = "\n\n".join(f"Problem{rep_text(idx)}: {problem_text}" for idx in range(num_repeats))

    if filler_tokens is not None:
        if filler_type == "numbers":
            filler = " ".join(str(i) for i in range(1, filler_tokens + 1))
        elif filler_type == "ellipsis":
            filler = " ".join(["..."] * filler_tokens)
        elif filler_type == "lorem":
            words = []
            for i in range(filler_tokens):
                words.append(LOREM_IPSUM_WORDS[i % len(LOREM_IPSUM_WORDS)])
            filler = " ".join(words)
        else:
            raise ValueError(f"Unknown filler_type: {filler_type}")
        out += f"\n\nFiller: {filler}"

    return out


def build_gemini_messages(
    few_shot_problems, problem_text, repeat_problem: int | None = None, filler_tokens: int | None = None, filler_type: str = "numbers"
):
    """
    Build the messages list for OpenRouter API call (OpenAI chat format).

    Args:
        few_shot_problems: list of (index, problem_dict) tuples
        problem_text: the text of the problem to solve
        repeat_problem: number of times to repeat problems (None = 1)
        filler_tokens: number of filler tokens to add (None = none)
        filler_type: type of filler tokens ('numbers', 'ellipsis', or 'lorem')

    Returns:
        list of message dicts in OpenAI chat format
    """
    system_instruction = get_gemini_system_instruction(filler_tokens, filler_type)
    messages = [{"role": "system", "content": system_instruction}]

    # Add few-shot examples
    for idx, prob in few_shot_problems:
        formatted_problem = build_problem_text(prob["problem"], repeat_problem, filler_tokens, filler_type)
        messages.append({"role": "user", "content": formatted_problem + EXTRA_APPEND})
        messages.append({"role": "assistant", "content": f"Answer: {prob['answer']}"})

    # Add the actual problem
    formatted_problem = build_problem_text(problem_text, repeat_problem, filler_tokens, filler_type)
    messages.append({"role": "user", "content": formatted_problem + EXTRA_APPEND})

    return messages


def get_gemini_cache_key(model, messages):
    """
    Generate a cache key for Gemini requests.

    Uses the messages directly since they're in OpenAI format (JSON serializable).
    """
    return {
        "model": model,
        "type": "gemini_openrouter_response",
        "version": "gemini_openrouter_v3",  # Bump this if prompt structure changes
        "messages": messages,
    }


def get_validation_cache_key(thinking_text, problem_text):
    """
    Generate a cache key for validation requests.
    Uses a hash of the thinking text to keep the key manageable.
    """
    thinking_hash = hashlib.sha256(thinking_text.encode()).hexdigest()[:16]
    return {
        "type": "gemini_validation",
        "version": "validation_v1",
        "thinking_hash": thinking_hash,
        "problem": problem_text,
    }


def parse_validation_response(raw_response):
    """
    Parse the validation response from Claude.

    Returns:
        tuple of (reasoned_about_details: bool, basically_same_as_example: bool)
    """
    try:
        # Find JSON in response
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw_response[start:end]
            parsed = json.loads(json_str)
            reasoned = parsed.get("reasoned_about_details", "").upper() == "YES"
            same_as_example = parsed.get("basically_same_as_example_thinking", "").upper() == "YES"
            return reasoned, same_as_example
        else:
            # Fallback: try to find YES/NO patterns
            reasoned = "YES" in raw_response.upper() and "reasoned" in raw_response.lower()
            return reasoned, False
    except (json.JSONDecodeError, KeyError):
        # Conservative fallback: assume it reasoned if we can't parse
        return True, False


async def check_gemini_reasoning(thinking_text, problem_text, anthropic_client, response_cache, timeout=120.0):
    """
    Use Claude to check if Gemini actually reasoned about the problem.
    Results are cached.

    Args:
        thinking_text: The thinking/reasoning text from Gemini
        problem_text: The math problem text
        anthropic_client: AsyncAnthropic client
        response_cache: ResponseCache instance for caching
        timeout: Timeout in seconds

    Returns:
        dict with keys:
            - reasoned_about_details: bool (True if Gemini reasoned about the problem)
            - basically_same_as_example: bool (True if thinking matches expected template)
            - is_valid: bool (True if response should be accepted)
            - raw_response: str (the raw response from Claude)
            - cached: bool (True if result was from cache)
    """
    # Check cache first
    cache_key = get_validation_cache_key(thinking_text, problem_text)
    cached_result = await response_cache.get(cache_key)

    if cached_result:
        return {
            "reasoned_about_details": cached_result["reasoned_about_details"],
            "basically_same_as_example": cached_result["basically_same_as_example"],
            "is_valid": cached_result["is_valid"],
            "raw_response": cached_result.get("raw_response", ""),
            "cached": True,
        }

    check_prompt = f"""Here is some thinking/reasoning done by a different AI system:
<thinking>
{thinking_text.strip()}
</thinking>

Did this other AI reason about the details of the following math problem at all?
<problem>
{problem_text.strip()}
</problem>

Is the thinking the AI did basically the same as this example thinking (e.g. it looks like a paraphrase of this example thinking)?
<example_thinking>
{EMPTY_THINKING_FOR_GEMINI.strip()}
</example_thinking>

Ok, you should answer these two questions. Here is the thinking done by this other AI system again:
<thinking>
{thinking_text.strip()}
</thinking>

Now, after thinking about it, answer only with "YES" or "NO" to the following question:
1. Did this other AI reason about the details of the following math problem at all?
2. Is the thinking the AI did basically the same as this example thinking (e.g. it looks like close to a paraphrase of this example thinking)?

Your output should be in this format:
{{
    "reasoned_about_details": "[YES/NO]",
    "basically_same_as_example_thinking": "[YES/NO]"
}}

(With [YES/NO] replaced with your actual answers.)"""

    # print("="*40)
    # print("="*40)
    # print(check_prompt)
    # print("="*40)
    # print("="*40)

    max_retries = 10
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                anthropic_client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=6_000,
                    thinking={"type": "enabled", "budget_tokens": 4_000},
                    messages=[{"role": "user", "content": check_prompt}],
                    timeout=300.0,
                ),
                timeout=timeout,
            )
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, re-raise the exception
            delay = base_delay * min((2 ** attempt), 60.0)
            print(f"  Anthropic API error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    # Extract text response
    raw_response = ""
    for block in response.content:
        if block.type == "text":
            raw_response = block.text
            break

    # print(f"{raw_response=}")

    # Parse the response
    reasoned, same_as_example = parse_validation_response(raw_response)

    # Valid if: didn't reason about details AND thinking is similar to example
    is_valid = not reasoned and same_as_example

    result = {
        "reasoned_about_details": reasoned,
        "basically_same_as_example": same_as_example,
        "is_valid": is_valid,
        "raw_response": raw_response,
        "cached": False,
    }

    # Cache the result
    await response_cache.set(
        cache_key,
        {
            "reasoned_about_details": reasoned,
            "basically_same_as_example": same_as_example,
            "is_valid": is_valid,
            "raw_response": raw_response,
        },
    )

    return result


async def call_gemini_api(openrouter_client, messages, model):
    """
    Make the Gemini API call via OpenRouter with retry logic.

    Args:
        openrouter_client: AsyncOpenAI client configured for OpenRouter
        messages: list of message dicts in OpenAI chat format
        model: model name (will be converted to OpenRouter format)

    Returns:
        tuple of (answer_text, thinking_text)
    """
    max_retries = 10
    base_delay = 0.5

    # Convert model name to OpenRouter format
    if model == "gemini-3-pro-preview":
        openrouter_model = "google/gemini-3-pro-preview"
    elif model == "gemini-2.5-pro":
        openrouter_model = "google/gemini-2.5-pro-preview"
    elif model.startswith("google/"):
        openrouter_model = model
    else:
        openrouter_model = f"google/{model}"

    for attempt in range(max_retries):
        try:
            response = await openrouter_client.chat.completions.create(
                model=openrouter_model,
                messages=messages,
                extra_body={
                    "reasoning": {
                        "enabled": True,
                        "thinking_level": "low",
                        "max_tokens": 500,
                    }
                },
                temperature=0.2,  # a bit of randomness so resampling is useful (maybe?) but pretty low still
                max_tokens=1000,
            )
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, re-raise the exception
            delay = base_delay * min((2**attempt), 60.0)
            print(f"  Gemini API error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    # Extract response
    choice = response.choices[0]
    answer_text = choice.message.content or ""

    # Extract thinking from OpenRouter response
    thinking_text = ""
    if hasattr(choice.message, "reasoning") and choice.message.reasoning:
        thinking_text = choice.message.reasoning
    elif hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        thinking_text = choice.message.reasoning_content

    return answer_text, thinking_text


async def evaluate_gemini_problem(
    openrouter_client,
    anthropic_client,
    response_cache,
    few_shot_problems,
    problem,
    model="gemini-3-pro-preview",
    repeat_problem: int | None = None,
    filler_tokens: int | None = None,
    filler_type: str = "numbers",
    verbosity=2,
):
    """
    Evaluate a single problem using Gemini with validation.
    Caches both Gemini responses and validation results.

    Args:
        openrouter_client: AsyncOpenAI client configured for OpenRouter
        anthropic_client: AsyncAnthropic client (for validation)
        response_cache: ResponseCache instance for caching
        few_shot_problems: list of problem dicts (used for cache key)
        problem: dict with 'problem' key
        model: Gemini model to use
        repeat_problem: number of times to repeat problems (None = 1)
        filler_tokens: number of filler tokens to add (None = none)
        filler_type: type of filler tokens ('numbers', 'ellipsis', or 'lorem')
        verbosity: Verbosity level

    Returns:
        dict with keys:
            - response_text: The answer text from Gemini
            - thinking_text: The thinking text from Gemini
            - attempts: Number of attempts made
            - rejected_count: Number of rejected responses
            - validation_result: Final validation result
            - validation_passed: Whether validation passed
            - gemini_cached: Whether Gemini response was from cache
            - validation_cached: Whether validation was from cache
    """

    problem_text = problem["problem"]
    messages = build_gemini_messages(
        few_shot_problems, problem_text, repeat_problem=repeat_problem, filler_tokens=filler_tokens, filler_type=filler_type
    )

    # Check cache for Gemini response first (using messages directly as cache key)
    gemini_cache_key = get_gemini_cache_key(model, messages)
    cached_gemini = await response_cache.get(gemini_cache_key)

    gemini_cached = False
    answer_text = ""
    thinking_text = ""

    if cached_gemini:
        gemini_cached = True
        answer_text = cached_gemini["response_text"]
        thinking_text = cached_gemini["thinking_text"]

    overall_is_valid = False

    max_retries = 3
    retries = 0

    cleaned_answer_text = None
    validation = None

    while overall_is_valid is False and retries < max_retries:
        # if it is cached, we do the processing in this loop once and then exit at the end of the loop
        if not gemini_cached:
            answer_text, thinking_text = await call_gemini_api(openrouter_client, messages, model)

        cleaned_answer_text = (
            answer_text.replace(EMPTY_THINKING_FOR_GEMINI, "")
            .strip()
            .replace(EMPTY_THINKING_START, "")
            .strip()
            .lower()
            .removeprefix("[")
            .strip()
            .removeprefix("answer:")
            .removesuffix("]")
            .strip()
            .removesuffix("%")
            .strip()
            .removesuffix("%")
            .strip()
            .removesuffix("\\")
            .strip()
            .removesuffix("\\")
            .strip()
            .removesuffix(r"^\circ")
            .strip()
            .removesuffix(r"\circ")
            .strip()
            .replace(",", "")
        )

        try:
            predicted_answer = int(cleaned_answer_text)
            invalid_output = False
        except ValueError:
            if verbosity >= 1:
                print(
                    f"Gemini output is not a valid integer: cleaned output={cleaned_answer_text[:100]}"
                    + ("... [truncated]" if len(cleaned_answer_text) > 100 else "")
                )
            predicted_answer = -1
            invalid_output = True

        # Validate using Claude (also cached)
        validation = await check_gemini_reasoning(thinking_text, problem_text, anthropic_client, response_cache)

        overall_is_valid = validation["is_valid"] and not invalid_output

        if not overall_is_valid and verbosity >= 2:
            print(
                f"  Gemini validation failed: "
                f"reasoned={validation['reasoned_about_details']}, "
                f"same_as_example={validation['basically_same_as_example']}, "
                f"invalid_output={invalid_output}"
                f"{' [cached]' if gemini_cached else ''}"
            )

        retries += 1

        if gemini_cached:
            break

    # Cache the Gemini response
    if not gemini_cached:
        await response_cache.set(
            gemini_cache_key,
            {
                "response_text": answer_text,
                "thinking_text": thinking_text,
            },
        )

    assert validation is not None
    assert cleaned_answer_text is not None

    return {
        "response_text": cleaned_answer_text,
        "thinking_text": thinking_text,
        "attempts": retries,
        "rejected_count": max_retries if not overall_is_valid and gemini_cached else (retries - overall_is_valid),
        "validation_result": validation,
        "validation_passed": overall_is_valid,
        "gemini_cached": gemini_cached,
        "validation_cached": validation["cached"],
    }
