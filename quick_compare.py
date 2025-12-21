import sys
import json
from scipy import stats
import numpy as np

# path1, path2 = "eval_results/eval_results_opus3_r0.json", "eval_results/eval_results_opus3_r5.json"
path1, path2 = sys.argv[1], sys.argv[2]

with open(path1, "r") as f:
    data1 = json.loads(f.read())

with open(path2, "r") as f:
    data2 = json.loads(f.read())

assert [x["problem"] for x in data1["results"]] == [x["problem"] for x in data2["results"]]

correct1 = np.array([x["is_correct"] for x in data1["results"]])
correct2 = np.array([x["is_correct"] for x in data2["results"]])

improves = correct2 & (~correct1)
breaks = (~correct2) & correct1

print(f"{correct1.sum()=}/{len(correct1)}")
print(f"{correct2.sum()=}/{len(correct1)}")
print(f"{improves.sum()=}/{len(correct1)}")
print(f"{breaks.sum()=}/{len(correct1)}")

problems_improves = [x["problem"] for x, imp in zip(data1["results"], improves) if imp]

# for x in problems_improves:
#     print("==")
#     print(x)
#     print("==")
