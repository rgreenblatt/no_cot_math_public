#!/usr/bin/env python3
"""
Parse MathCounts problems from PDFs using Claude API.
Processes problem PDFs (Sprint, Target, Team, CDR) along with Answers.pdf
to extract problems with their answers into a structured JSON format.
"""

import json
import os
import asyncio
import base64
import argparse
import subprocess
from pathlib import Path
from anthropic import AsyncAnthropic
from response_cache import ResponseCache

# Minimum characters for a PDF to be considered to have extractable text
PDF_TEXT_THRESHOLD = 1000

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
response_cache = ResponseCache("caches/cache_mathcounts_parse.json")


def encode_pdf(pdf_path: str) -> str:
    """Base64 encode a PDF file."""
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def pdf_has_extractable_text(pdf_path: Path) -> bool:
    """
    Check if a PDF has extractable text (not just scanned images).
    Returns True if the PDF has more than PDF_TEXT_THRESHOLD characters of text.
    """
    try:
        result = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, timeout=10)
        char_count = len(result.stdout)
        # print(f"  PDF text character count for {pdf_path}: {char_count}")
        return char_count >= PDF_TEXT_THRESHOLD
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: Could not check PDF text for {pdf_path}: {e}")
        # If pdftotext fails or isn't available, assume the PDF is usable
        return True


def get_competition_pdfs(competition_dir: Path, check_text: bool = True) -> tuple[dict[str, Path], list[str]]:
    """
    Get relevant PDFs from a competition directory.

    Args:
        competition_dir: Path to the competition directory
        check_text: If True, check if PDFs have extractable text

    Returns:
        Tuple of (dict mapping PDF type to path, list of scan-only PDF names)
    """
    pdfs = {}
    scan_only = []
    relevant_files = ["Answers.pdf", "Sprint.pdf", "Target.pdf", "Team.pdf", "CDR.pdf"]

    for filename in relevant_files:
        pdf_path = competition_dir / filename
        if pdf_path.exists():
            pdf_type = filename.replace(".pdf", "")
            if check_text and not pdf_has_extractable_text(pdf_path):
                scan_only.append(pdf_type)
            else:
                pdfs[pdf_type] = pdf_path

    return pdfs, scan_only


async def parse_competition(
    year: str,
    round_name: str,
    pdfs: dict[str, Path],
    semaphore: asyncio.Semaphore,
    model: str = "claude-opus-4-5-20251101",
) -> list[dict]:
    """
    Parse problems from a single competition using Claude.

    Args:
        year: Competition year (e.g., "2015")
        round_name: Competition round (e.g., "CHAPTER", "STATE", "NATIONAL", "SCHOOL")
        pdfs: Dict mapping PDF type to file path
        semaphore: asyncio.Semaphore to limit concurrency
        model: Claude model to use

    Returns:
        List of problem dictionaries
    """
    if "Answers" not in pdfs:
        print(f"  Warning: No Answers.pdf found for {year}/{round_name}, skipping...")
        return []

    # Build the content array with all PDFs
    content = []

    # Add problem PDFs first (these are the competition types: Sprint, Target, Team, CDR)
    all_competition_types = ["Sprint", "Target", "Team", "CDR"]
    included_competition_types = []

    for comp_type in all_competition_types:
        if comp_type in pdfs:
            pdf_data = encode_pdf(str(pdfs[comp_type]))
            content.append(
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}}
            )
            content.append({"type": "text", "text": f"Above is the {comp_type} problems PDF."})
            included_competition_types.append(comp_type)

    if not included_competition_types:
        print(f"  Warning: No problem PDFs found for {year}/{round_name}, skipping...")
        return []

    # Add answers PDF
    answers_data = encode_pdf(str(pdfs["Answers"]))
    content.append(
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": answers_data}}
    )
    content.append({"type": "text", "text": "Above is the Answers PDF containing answers for all competition types."})

    # Add the extraction prompt
    prompt = f"""You are parsing MathCounts competition problems. This is the {year} {round_name} competition.

The included competition types are: {', '.join(included_competition_types)}

Please extract ALL problems from the problem PDFs and match them with their answers from the Answers PDF.

For each problem, output a JSON object with these fields:
- "problem": The full problem text (preserve any mathematical notation, use LaTeX where appropriate)
- "round": The competition round ("{round_name}" - this is SCHOOL, CHAPTER, STATE, or NATIONAL)
- "competition_type": The problem type (Sprint, Target, Team, or CDR)
- "problem_number": The problem number within that type (as an integer)
- "year": {year}
- "answer": The answer from the Answers PDF

Important notes:
- Sprint typically has 30 problems
- Target typically has 8 problems (presented in pairs)
- Team typically has 10 problems
- CDR (Countdown Round) problems may or may not be present
- Make sure to match answers correctly to their problems
- Preserve mathematical expressions accurately (use LaTeX notation like $x^2$ for superscripts, $\\frac{{a}}{{b}}$ for fractions, etc.)
- SKIP any problems that require a diagram/figure to solve. If a problem references "the figure", "the diagram", or contains an image that is essential to understanding the problem, do not include it.

Output ONLY a valid JSON array of problem objects, no other text. Example format:
[
  {{"problem": "What is $2 + 2$?", "round": "{round_name}", "competition_type": "Sprint", "problem_number": 1, "year": {year}, "answer": "4"}},
  ...
]"""

    content = [{"type": "text", "text": prompt}] + content

    messages = [{"role": "user", "content": content}]

    # Create cache key from the request parameters
    # We use file paths and model as the key (not the full PDF data for efficiency)
    cache_key = {
        "prompt": prompt,
        "model": model,
        "year": year,
        "round": round_name,
        "pdfs": {k: str(v) for k, v in pdfs.items()},
        "competition_types": included_competition_types,
    }

    # Check cache first (outside semaphore)
    cached_response = await response_cache.get(cache_key)

    async with semaphore:
        if cached_response:
            print(f"[CACHED] {year}/{round_name} ({', '.join(included_competition_types)})")
            response_text = cached_response.get("response", "")
        else:
            print(f"Sending request to Claude for {year}/{round_name} ({', '.join(included_competition_types)})...")

            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=16000,
                        messages=messages,
                        timeout=60 * 15,
                    ),
                    timeout=60 * 20,
                )

                # Extract response text
                response_text = response.content[0].text.strip()

                # Cache the response
                await response_cache.set(cache_key, {"response": response_text})

            except asyncio.TimeoutError:
                print(f"TIMEOUT on {year}/{round_name} after 6 minutes")
                return []

        # Try to parse as JSON
        # Sometimes the model wraps in ```json ... ```
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        problems = json.loads(response_text)

        # Validate and fix year/round/competition_type in returned problems
        # round = SCHOOL/CHAPTER/STATE/NATIONAL
        # competition_type = Sprint/Target/Team/CDR
        for i, p in enumerate(problems):
            expected_year = int(year)
            if p.get("year") != expected_year:
                print(f"  Warning: Problem {i+1} has year {p.get('year')}, expected {expected_year}. Fixing.")
                p["year"] = expected_year
            if p.get("round") != round_name:
                print(f"  Warning: Problem {i+1} has round '{p.get('round')}', expected '{round_name}'. Fixing.")
                p["round"] = round_name
            if p.get("competition_type") not in included_competition_types:
                print(
                    f"  Warning: Problem {i+1} has competition_type '{p.get('competition_type')}' which is not in {included_competition_types}"
                )

        print(f"  Extracted {len(problems)} problems from {year}/{round_name}")
        return problems


def find_all_competitions(base_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Find all competition directories.
    Returns list of (year, round_name, path) tuples.
    """
    competitions = []

    for year_dir in sorted(base_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        year = year_dir.name

        for comp_dir in sorted(year_dir.iterdir()):
            if not comp_dir.is_dir():
                continue

            round_name = comp_dir.name
            competitions.append((year, round_name, comp_dir))

    return competitions


async def run_parsing(
    input_dir: str,
    output_file: str,
    model: str = "claude-opus-4-5-20251101",
    concurrency: int = 5,
    year_filter: str | None = None,
    round_filter: str | None = None,
    dry_run: bool = False,
):
    """
    Run parsing on all competitions.

    Args:
        input_dir: Base directory containing year folders with PDFs
        output_file: Output JSONL file path
        model: Claude model to use
        concurrency: Maximum concurrent API calls
        year_filter: Process only this year (optional)
        round_filter: Process only this round (SCHOOL, CHAPTER, STATE, NATIONAL) (optional)
        dry_run: If True, only show what would be processed without making API calls
    """
    base_dir = Path(input_dir)
    if not base_dir.exists():
        print(f"Error: Input directory '{base_dir}' does not exist")
        return

    # Find all competitions
    competitions = find_all_competitions(base_dir)

    # Filter if specific year/round requested
    if year_filter:
        competitions = [(y, r, p) for y, r, p in competitions if y == year_filter]
    if round_filter:
        competitions = [(y, r, p) for y, r, p in competitions if r == round_filter]

    print(f"Found {len(competitions)} competitions to process")
    print(f"Concurrency: {concurrency}")
    print(f"Model: {model}")

    if dry_run:
        print("\n[DRY RUN] Would process the following competitions:")
        total_pdfs = 0
        skipped_count = 0
        for year, round_name, comp_dir in competitions:
            pdfs, scan_only = get_competition_pdfs(comp_dir)
            if scan_only:
                print(f"  {year}/{round_name}: SKIPPED (scan-only: {', '.join(scan_only)})")
                skipped_count += 1
            elif pdfs:
                comp_types = [k for k in pdfs.keys() if k != "Answers"]
                print(f"  {year}/{round_name}: {', '.join(comp_types)}")
                total_pdfs += len(pdfs)
        print(f"\n[DRY RUN] Total: {len(competitions) - skipped_count} competitions, {total_pdfs} PDFs")
        print(f"[DRY RUN] Skipped: {skipped_count} competitions with scan-only PDFs")
        print("[DRY RUN] No API calls made.")
        return

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all competitions
    tasks = []
    skipped_competitions = []
    for year, round_name, comp_dir in competitions:
        pdfs, scan_only = get_competition_pdfs(comp_dir)
        if scan_only:
            skipped_competitions.append((year, round_name, scan_only))
            print(f"Skipping {year}/{round_name} (scan-only PDFs: {', '.join(scan_only)})")
        elif pdfs:
            tasks.append(
                parse_competition(
                    year=year,
                    round_name=round_name,
                    pdfs=pdfs,
                    semaphore=semaphore,
                    model=model,
                )
            )

    if skipped_competitions:
        print(f"\nSkipped {len(skipped_competitions)} competitions with scan-only PDFs")

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_problems = []
    for problem_list in results:
        all_problems.extend(problem_list)

    # Save any remaining cached responses
    await response_cache.save_cache(force=True)

    # Write output as JSONL
    print(f"\nWriting {len(all_problems)} problems to {output_file}...")
    with open(output_file, "w") as f:
        for problem in all_problems:
            f.write(json.dumps(problem) + "\n")

    print("Done!")

    # Print summary
    by_round = {}
    by_year = {}
    by_comp_type = {}
    for p in all_problems:
        by_round[p["round"]] = by_round.get(p["round"], 0) + 1
        by_year[p["year"]] = by_year.get(p["year"], 0) + 1
        by_comp_type[p["competition_type"]] = by_comp_type.get(p["competition_type"], 0) + 1

    print("\nSummary:")
    print(f"  Total problems: {len(all_problems)}")
    print(f"  By round: {by_round}")
    print(f"  By year: {by_year}")
    print(f"  By competition type: {by_comp_type}")


def main():
    parser = argparse.ArgumentParser(description="Parse MathCounts problems from PDFs")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="./mathcounts_pdfs",
        help="Base directory containing year folders with PDFs",
    )
    parser.add_argument("--output", "-o", type=str, default="mathcounts_problems.jsonl", help="Output JSONL file path")
    parser.add_argument("--model", "-m", type=str, default="claude-opus-4-5-20251101", help="Claude model to use")
    parser.add_argument("--year", "-y", type=str, default=None, help="Process only a specific year")
    parser.add_argument(
        "--round", "-r", type=str, default=None, help="Process only a specific round (SCHOOL, CHAPTER, STATE, NATIONAL)"
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=5, help="Number of concurrent API requests (default: 5)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be processed without making API calls"
    )

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output file: {args.output}")
    print(f"  Model: {args.model}")
    print(f"  Concurrency: {args.concurrency}")
    if args.year:
        print(f"  Year filter: {args.year}")
    if args.round:
        print(f"  Round filter: {args.round}")
    if args.dry_run:
        print(f"  Dry run: True")

    asyncio.run(
        run_parsing(
            input_dir=args.input_dir,
            output_file=args.output,
            model=args.model,
            concurrency=args.concurrency,
            year_filter=args.year,
            round_filter=args.round,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
