"""Write deterministic first-1000 Alpaca CSVs for the benign-task LoRA run."""
from safeoff.paths import project_root

import csv
import hashlib
import json
from pathlib import Path


ROOT = project_root()
OUTPUT_DIR = ROOT / "data/benign_task_1000"
SOURCES = {
    "gsm8k_0_999": ROOT / "data/gsm8k_as_alpaca.csv",
    "samsum_0_999": ROOT / "data/samsum_as_alpaca.csv",
}
COUNT = 1000
FIELDS = ["index", "instruction", "input", "alpaca_reference", "suffix"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection": "first 1000 rows in source CSV order",
        "count_per_dataset": COUNT,
        "datasets": {},
    }
    for name, source in SOURCES.items():
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))[:COUNT]
        if len(rows) != COUNT:
            raise ValueError("{} has only {} rows".format(source, len(rows)))
        if len({row["index"] for row in rows}) != COUNT:
            raise ValueError("{} does not have 1000 unique selected indices".format(source))
        if any(not row["instruction"].strip() or not row["alpaca_reference"].strip() for row in rows):
            raise ValueError("{} contains an empty instruction or response".format(source))
        output = OUTPUT_DIR / (name + ".csv")
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "index": row["index"],
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "alpaca_reference": row["alpaca_reference"],
                    "suffix": "",
                })
        manifest["datasets"][name] = {
            "source": str(source.relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)),
            "first_source_index": rows[0]["index"],
            "last_source_index": rows[-1]["index"],
            "sha256": sha256(output),
        }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
