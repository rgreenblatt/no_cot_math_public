import json
import random
import numpy as np

random.seed(42)  # For reproducibility
np.random.seed(42)

operators = ["+", "-", "*", "//", "%"]
op_probs = [0.3, 0.3, 0.15, 0.1, 0.05]
op_probs = np.array(op_probs)
op_probs = op_probs / op_probs.sum()


def rand_int():
    return random.randint(-99, 99)


def rand_int_for_div_mod():
    val = random.randint(-9, 9)
    while val == 0:
        val = random.randint(-9, 9)
    return val


def sample_expr(is_top_level=False, num_branches=None):
    if num_branches is None:
        assert is_top_level
        # num_branches = random.randint(1, 5)
        num_branches = random.randint(5, 7)

    if num_branches == 0:
        return str(rand_int())

    num_branches -= 1
    left_branches = random.randint(0, num_branches)
    right_branches = num_branches - left_branches

    l = sample_expr(num_branches=left_branches)
    r = sample_expr(num_branches=right_branches)

    op = np.random.choice(operators, p=op_probs)
    if op in ["//", "%"]:
        while eval(r) == 0:
            r = sample_expr(num_branches=right_branches)
    if is_top_level:
        return f"{l} {op} {r}"
    return f"({l} {op} {r})"


# Generate problems with varying difficulty
problems = [sample_expr(is_top_level=True) for _ in range(3000)]

with open("arithmetic_problems.jsonl", "w") as file:
    for i, prob in enumerate(problems):
        answer = eval(prob)
        problem_entry = {
            "problem": f"Evaluate this Python expression. {prob}",
            "round": "arithmetic",
            "category": "n/a",
            "problem_number": i,
            "answer": answer,
        }
        file.write(json.dumps(problem_entry) + "\n")
