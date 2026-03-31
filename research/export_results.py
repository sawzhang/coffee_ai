"""
Export research results to site/data/ for GitHub Pages visualization.

Reads results.tsv and outputs:
- site/data/results.json  — experiment history
- site/data/beans_summary.json — bean dataset summary for the site
"""

import json
import csv
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
SITE_DATA_DIR = RESEARCH_DIR.parent / "site" / "data"


def export_results():
    """Convert results.tsv to results.json."""
    tsv_path = RESEARCH_DIR / "results.tsv"
    if not tsv_path.exists():
        print("No results.tsv found, creating empty results.json")
        results = []
    else:
        results = []
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                results.append({
                    "id": i + 1,
                    "commit": row.get("commit", ""),
                    "val_mae": float(row.get("val_mae", 0)),
                    "num_params": int(row.get("num_params", 0)),
                    "status": row.get("status", ""),
                    "description": row.get("description", ""),
                })

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SITE_DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported {len(results)} experiments to {SITE_DATA_DIR / 'results.json'}")


def export_beans_summary():
    """Export a lightweight bean summary for the site."""
    beans_path = RESEARCH_DIR / "data" / "beans.json"
    if not beans_path.exists():
        print("No beans.json found")
        return

    with open(beans_path) as f:
        beans = json.load(f)

    # Summary stats
    scores = [b["scores"]["overall"] for b in beans]
    countries = {}
    varieties = {}
    methods = {}
    for b in beans:
        c = b["G"]["country"]
        countries[c] = countries.get(c, 0) + 1
        v = b["G"]["variety"]
        varieties[v] = varieties.get(v, 0) + 1
        m = b["P"]["method"]
        methods[m] = methods.get(m, 0) + 1

    summary = {
        "total_beans": len(beans),
        "score_range": [round(min(scores), 1), round(max(scores), 1)],
        "score_mean": round(sum(scores) / len(scores), 1),
        "countries": dict(sorted(countries.items(), key=lambda x: -x[1])),
        "varieties": dict(sorted(varieties.items(), key=lambda x: -x[1])),
        "processing_methods": dict(sorted(methods.items(), key=lambda x: -x[1])),
        "top_beans": sorted(
            [{"name": b["name"], "score": b["scores"]["overall"],
              "country": b["G"]["country"], "variety": b["G"]["variety"],
              "process": b["P"]["method"], "roast": b["R"]["roast_level"]}
             for b in beans],
            key=lambda x: -x["score"]
        )[:20],
    }

    with open(SITE_DATA_DIR / "beans_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Exported bean summary to {SITE_DATA_DIR / 'beans_summary.json'}")


if __name__ == "__main__":
    export_results()
    export_beans_summary()
