import argparse
import json
from pathlib import Path

from models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="out/results.json")
parser.add_argument("--out", default="out/report.html")
args = parser.parse_args()

results = json.loads(Path(args.results).read_text())

# Results from before the runner recorded input descriptions: fill them in from the current suite.
labels = {m["name"]: m["input"].label for m in MODELS}
for row in results["runs"]:
    if "input" not in row and row["model"] in labels:
        row["input"] = labels[row["model"]]

template = (Path(__file__).parent / "report_template.html").read_text()
embedded = json.dumps(results).replace("</", "<\\/")
Path(args.out).write_text(template.replace("__RESULTS_JSON__", embedded))
print(f"wrote {args.out}")
