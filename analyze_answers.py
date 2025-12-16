#!/usr/bin/env python3
"""
Analyze answer types in mathcounts_problems.jsonl.
Reports what fraction of problems have integer, fraction, or other answer types.
"""

import json
import re
import argparse
from collections import defaultdict


def classify_answer(answer: str) -> tuple[str, str]:
    """
    Classify an answer as 'integer', 'fraction', or 'other'.
    Returns (type, normalized_answer).
    """
    if not answer:
        return "other", answer

    original = answer
    answer = str(answer).strip()

    # Remove common units and suffixes
    units = [
        # Length
        r"\s*(cm|mm|m|km|in|ft|yd|mi|inches|feet|yards|miles|meters|centimeters|millimeters|kilometers)\s*$",
        # Area
        r"\s*(cm\^?2|m\^?2|ft\^?2|in\^?2|sq\.?\s*(cm|m|ft|in|units?)?|square\s+(cm|m|ft|in|units?))\s*$",
        # Volume
        r"\s*(cm\^?3|m\^?3|ft\^?3|in\^?3|cubic\s+(cm|m|ft|in|units?))\s*$",
        # Time
        r"\s*(sec|seconds?|min|minutes?|hrs?|hours?|days?|weeks?|months?|years?)\s*$",
        # Money
        r"^\$\s*",
        r"\s*(dollars?|cents?)\s*$",
        # Percent
        r"\s*%\s*$",
        r"\s*percent\s*$",
        # Degrees
        r"\s*°\s*$",
        r"\s*degrees?\s*$",
        # Other common units
        r"\s*(mph|km/h|m/s|ft/s)\s*$",
        r"\s*(lbs?|pounds?|kg|kilograms?|g|grams?|oz|ounces?)\s*$",
        r"\s*(L|liters?|ml|milliliters?|gal|gallons?)\s*$",
        r"\s*units?\s*$",
        r"\s*people\s*$",
        r"\s*students?\s*$",
        r"\s*ways?\s*$",
        r"\s*points?\s*$",
        r"\s*games?\s*$",
        r"\s*problems?\s*$",
    ]

    for unit_pattern in units:
        answer = re.sub(unit_pattern, "", answer, flags=re.IGNORECASE)

    answer = answer.strip()

    # Remove commas from numbers
    answer = answer.replace(",", "")

    # Check for pure integer (possibly negative)
    if re.match(r"^-?\d+$", answer):
        return "integer", answer

    # Check for decimal that equals an integer
    if re.match(r"^-?\d+\.0+$", answer):
        return "integer", answer.split(".")[0]

    # Check for fraction (a/b)
    fraction_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", answer)
    if fraction_match:
        return "fraction", answer

    # Check for mixed number (a b/c or a-b/c)
    mixed_match = re.match(r"^(-?\d+)\s*[-\s]\s*(\d+)\s*/\s*(\d+)$", answer)
    if mixed_match:
        return "fraction", answer

    # Check for decimal
    if re.match(r"^-?\d+\.\d+$", answer):
        return "decimal", answer

    # Check for scientific notation
    if re.match(r"^-?\d+(\.\d+)?[eE][+-]?\d+$", answer):
        return "scientific", answer

    return "other", original


def main():
    parser = argparse.ArgumentParser(description="Analyze answer types in mathcounts problems")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="mathcounts_problems.jsonl",
        help="Input JSONL file (default: mathcounts_problems.jsonl)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show examples of each answer type")

    args = parser.parse_args()

    # Load problems
    problems = []
    with open(args.input, "r") as f:
        for line in f:
            problems.append(json.loads(line))

    # Classify answers
    type_counts = defaultdict(int)
    type_examples = defaultdict(list)

    for p in problems:
        answer = p.get("answer", "")
        answer_type, normalized = classify_answer(answer)
        type_counts[answer_type] += 1

        if len(type_examples[answer_type]) < 10:
            type_examples[answer_type].append((answer, normalized))

    total = len(problems)

    print(f"Total problems: {total}\n")
    print("Answer type breakdown:")
    print("-" * 40)

    for answer_type in ["integer", "fraction", "decimal", "scientific", "other"]:
        count = type_counts[answer_type]
        if count > 0:
            pct = 100 * count / total
            print(f"  {answer_type:12}: {count:5} ({pct:5.1f}%)")

    print("-" * 40)

    if args.verbose:
        print("\nExamples of each type:")
        for answer_type in ["integer", "fraction", "decimal", "scientific", "other"]:
            if type_examples[answer_type]:
                print(f"\n{answer_type.upper()}:")
                for original, normalized in type_examples[answer_type]:
                    if original != normalized:
                        print(f"  '{original}' -> '{normalized}'")
                    else:
                        print(f"  '{original}'")


if __name__ == "__main__":
    main()
