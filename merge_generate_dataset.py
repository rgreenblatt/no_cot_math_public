# %%
import json
import itertools
import random
import re
import pandas as pd
from datasets import load_dataset
from analyze_answers import classify_answer


def get_last_number(s):
    numbers = re.findall(r"-?\d+\.?\d*", s)
    if not numbers:
        return None
    num_str = numbers[-1]
    # Only treat as float if there are digits after the decimal
    if "." in num_str and num_str.split(".")[1]:
        return float(num_str)
    return int(num_str.rstrip("."))


def is_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


with open("durer_problems.jsonl", "r") as file:
    probs = [json.loads(line) for line in file]

with open("kavicskupa_2024.jsonl", "r") as file:
    probs_extra = [json.loads(line) for line in file]

with open("hmmt_problems.jsonl", "r") as file:
    probs_hmmt = [json.loads(line) for line in file]
    probs_hmmt_out = [
        {
            **p,
            "category": "N/A",
            "solution": f"Answer: {p['answer']}",
            "answer": int(p["answer"]),
        }
        for p in probs_hmmt
        if is_int(p["answer"])
    ]

with open("kavicskupa_problems_2013_2018.jsonl", "r") as file:
    probs_kavics_next = [json.loads(line) for line in file]
    probs_kavics_next = [{**p, "solution": f"Answer: {p['answer']}"} for p in probs_kavics_next]

with open("kavicskupa_problems_2019_2023.jsonl", "r") as file:
    probs_kavics_next_next = [json.loads(line) for line in file]
    probs_kavics_next_next = [
        {**p, "solution": f"Answer: {p['answer']}"}
        for p in probs_kavics_next_next  # if p["round"] == "kavicskupa_2023_en"
    ]

# AMC is probably contaminated, not extremely easy to tell, but probably.

# art_of_problem_solving = pd.read_csv("parsed_ArtOfProblemSolving.csv") # from kaggle dataset
# amc_probs = art_of_problem_solving[art_of_problem_solving["link"].str.contains("AMC", na=False)]
# amc_8_probs = art_of_problem_solving[art_of_problem_solving["link"].str.contains("AMC_8", na=False)]
# amc_8_probs_integers = amc_8_probs[amc_8_probs["answer"].apply(lambda x: is_int(x))]

# amc_8_dicts = [{"category": x["letter"], "round": f"AMC_8", "problem_number": i, "problem": x["problem"], "solution": x["solution"], "answer": int(x["answer"])} for i,(_, x) in enumerate(amc_8_probs_integers.iterrows())]
# # random.seed(42)
# # random.shuffle(amc_8_dicts)

with open("mathcounts_problems.jsonl", "r") as file:
    probs_mathcounts_initial = [json.loads(line) for line in file]
    probs_mathcounts = []
    for p in probs_mathcounts_initial:
        answer_type, ans = classify_answer(p["answer"])
        if answer_type != "integer":
            continue
        if p["competition_type"].lower() not in ["target", "team", "sprint"]:
            continue
        # if p["round"].lower() not in ["state"]:
        #     continue
        if p["round"].lower() not in ["national", "state"]:
            continue
        out_dict = {
            **p,
            "category": "n/a",
            "round": "mathcounts-" + p["round"].lower() + "-" + p["competition_type"].lower(),
            "answer": int(ans),
            "solution": f"Answer: {p['answer']}",
        }
        del out_dict["competition_type"]
        probs_mathcounts.append(out_dict)


# this order probably gets you slightly nicer few shot examples?
probs = probs + probs_extra + probs_mathcounts + probs_hmmt_out + probs_kavics_next_next + probs_kavics_next
# probs = probs_hmmt_out
# probs = amc_8_dicts[:300]
# probs = probs_mathcounts

with open("durer_answers.json", "r") as file:
    ans = json.loads(file.read())

# print(get_last_number("abc 123 def 456"))

# %%

# len(probs_mathcounts)


# %%

cat_num_round_to_table_ans = {}

for a in ans:
    cat, num = a["problem_id"].split("-")
    cat_num_round_to_table_ans[(cat, int(num), a["round"])] = int(a["ans"])

# %%

probs_num_initial = []

for p in probs:
    if "For the solution" in p["solution"] and "Answer:" not in p["solution"]:
        num_sol = None
    else:
        num_sol = get_last_number(p["solution"])
        # if num_sol is None:
        #     continue

    probs_num_initial.append({**p, "sol_num": num_sol})

# %%


cat_num_round_to_prob = {(p["category"], p["problem_number"], p["round"]): p for p in probs_num_initial}


# %%

probs_num = []

for p in probs_num_initial:

    if "For the solution" in p["solution"] and "Answer:" not in p["solution"]:
        assert p["solution"].startswith("For the solution, see Category ")
        rem = p["solution"].removeprefix("For the solution, see Category ")
        cat = rem[0]
        rem = rem[2:].strip()
        assert rem.startswith("Problem ")
        rem = rem.removeprefix("Problem ")
        assert rem.endswith(".")
        prob_num = int(rem[:-1])

        num_sol = cat_num_round_to_prob[(cat, prob_num, p["round"])]["sol_num"]

        probs_num.append({**p, "sol_num": num_sol})

    else:
        probs_num.append(p)

    # print(get_last_number(p["solution"]))

# %%

probs_with_answers = []

already_existing_probs = {}

for p in probs_num:
    # if p["round"] ==

    if p["problem"].startswith("Game:"):
        continue

    conv_round = {
        "Regional round": "regional",
        "Online round": "online",
        "Final round - day 1": "final-day1",
        "Final round - day 2": "final-day2",
        "Kavicskupa 2024": "kavicskupa2024",
    }

    if p["round"] in conv_round:
        round_conv = conv_round[p["round"]]

        if (round_conv == "final-day1") or (round_conv == "regional" and p["category"] in ["C", "D", "E", "E+"]):
            continue
    else:
        round_conv = p["round"]

    if p["answer"] is None:
        assert not round_conv.startswith("kavic")
        assert round_conv in conv_round.values()
        table_ans = cat_num_round_to_table_ans[(p["category"], p["problem_number"], round_conv)]

        new_ans = int(table_ans)
    else:
        new_ans = int(p["answer"])

    if (
        "diagram" in p["problem"].lower()
        or "figure" in p["problem"].lower()
        or "fill in the grid" in p["problem"].lower()
        or "in her grid notebook" in p["problem"].lower()
    ) and "kavicskupa" not in p["round"]:
        continue

    truncated_prob = p["problem"][:140]

    if truncated_prob in already_existing_probs:
        if p["problem"] != already_existing_probs[truncated_prob]["problem"]:
            print(
                "\n\nDIFFERENT problem text, same prefix:",
                p["problem"],
                already_existing_probs[truncated_prob]["problem"] + "\n\n",
                sep="\n---\n",
            )
        continue

    else:
        already_existing_probs[truncated_prob] = p

    out = {**p, "answer": new_ans}
    del out["sol_num"]
    del out["solution"]

    probs_with_answers.append(out)

    # if p["answer"] is None or int(p["answer"]) != table_ans:
    #     print(p, table_ans)

len(probs_with_answers)

# %%

# AIME is massively contaminated for opus lol

# probs_with_answers_just_aime = []

# # Load AIME dataset from Hugging Face
# aime_dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")

# for item in aime_dataset:
#     year = item["Year"]
#     problem_number = item["Problem Number"]
#     part = item["Part"]
#     question = item["Question"]
#     answer = item["Answer"]

#     try:
#         answer = int(answer)
#     except ValueError:
#         print(
#             f"Skipping AIME problem {year} {part} {problem_number} due to non-integer answer: {answer}"
#         )
#         continue

#     # Format round as AIME-YEAR-PART
#     round_name = f"AIME-{year}" + (f"-{part}" if part is not None else "")

#     # Create problem entry in the same format as probs_with_answers
#     aime_problem = {
#         "category": "N/A",
#         "problem_number": problem_number,
#         "round": round_name,
#         "problem": question,
#         "answer": int(answer),
#     }

#     probs_with_answers_just_aime.append(aime_problem)

# probs_with_answers_just_aime = sorted(
#     probs_with_answers_just_aime, key=lambda x: x["problem_number"]
# )

# with open("aime_problems_with_answers.jsonl", "w") as file:
#     for p in probs_with_answers_just_aime:
#         file.write(json.dumps(p) + "\n")

# probs_with_answers_with_aime = probs_with_answers + probs_with_answers_just_aime

# len(probs_with_answers_with_aime)

# %%

# MATH also is massively contaminated for opus lol

# probs_with_answers_just_math = []

# # Load MATH dataset from Hugging Face
# math_datasets = [
#     load_dataset("EleutherAI/hendrycks_math", subset, split="test")
#     for subset in [
#         "algebra",
#         "counting_and_probability",
#         "geometry",
#         "intermediate_algebra",
#         "number_theory",
#         "prealgebra",
#         "precalculus",
#     ]
# ]

# items = [item for ds in math_datasets for item in ds]


# for i, item in enumerate(items):
#     problem_number = i
#     category = item["level"]
#     round_name = "MATH" + "-" + item["type"].lower().replace(" ", "_")
#     question = item["problem"]

#     solution = item["solution"]
#     assert "\\boxed{" in solution
#     parse_out_boxed = re.search(r"\\boxed\{([^}]*)\}", solution)
#     assert parse_out_boxed is not None
#     answer_str = parse_out_boxed.group(1)
#     try:
#         answer = int(answer_str)
#     except ValueError:
#         print(
#             f"Skipping MATH problem {category} {round_name} {problem_number} due to non-integer answer: {answer_str}"
#         )
#         continue

#     # try:
#     #     answer = int(answer)
#     # except ValueError:
#     #     print(
#     #         f"Skipping problem {year} {part} {problem_number} due to non-integer answer: {answer}"
#     #     )
#     #     continue

#     # Format round as AIME-YEAR-PART

#     # Create problem entry in the same format as probs_with_answers
#     math_problem = {
#         "category": category,
#         "problem_number": problem_number,
#         "round": round_name,
#         "problem": question,
#         "answer": int(answer),
#     }

#     probs_with_answers_just_math.append(math_problem)

# len(probs_with_answers_just_math)

# probs_with_answers_just_math = sorted(
#     probs_with_answers_just_math, key=lambda x: x["category"]
# )

# with open("MATH_problems_with_answers.jsonl", "w") as file:
#     for p in probs_with_answers_just_math:
#         file.write(json.dumps(p) + "\n")

# probs_with_answers_with_math = probs_with_answers_just_math + probs_with_answers

# len(probs_with_answers_with_math)

# %%

len(probs_with_answers)


# %%

with open("problems_with_answers.jsonl", "w") as file:
    for p in probs_with_answers:
        file.write(json.dumps(p) + "\n")
