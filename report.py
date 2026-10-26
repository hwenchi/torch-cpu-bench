import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="out/results.json")
parser.add_argument("--out", default="out/report.html")
args = parser.parse_args()

results = json.loads(Path(args.results).read_text())
template = (Path(__file__).parent / "report_template.html").read_text()
embedded = json.dumps(results).replace("</", "<\\/")
Path(args.out).write_text(template.replace("__RESULTS_JSON__", embedded))
print(f"wrote {args.out}")
