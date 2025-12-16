import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.loads(f.read())

acc = data["summary"]["accuracy"]
correct = data["summary"]["correct"]
count = data["summary"]["total"]

print(f"Accuracy: {acc:.2%} ({correct}/{count})")
