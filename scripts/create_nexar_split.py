import csv
import json
import hashlib
import random
from pathlib import Path
from collections import Counter

ROOT = Path("data/nexar-collision-prediction")
SOURCE = ROOT / "train.csv"
OUT = ROOT / "splits"

SEED = 42
SPLIT_SIZE = 375

SMOKE_IDS = {
    "00072", "00128", "00205", "00208", "00408",
    "00457", "00471", "00822", "00900", "00948",
    "01171", "01267", "01286", "01429", "01486",
    "01518", "01802", "01889", "01904", "01924"
}

# Never overwrite an existing split.
if any((OUT / name).exists() for name in
       ["dev.csv", "heldout.csv", "split_manifest.json"]):
    raise SystemExit(
        "Split files already exist. Inspect them before proceeding."
    )

if not SOURCE.exists():
    raise SystemExit(f"Missing source file: {SOURCE}")

source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

# Normalize IDs and validate input.
for row in rows:
    row["id"] = row["id"].strip().zfill(5)
    row["target"] = str(int(row["target"]))

ids = [r["id"] for r in rows]

if len(ids) != len(set(ids)):
    raise SystemExit("Duplicate video IDs found.")

if len(rows) != 1500:
    raise SystemExit(f"Expected 1500 rows, found {len(rows)}.")

if set(r["target"] for r in rows) != {"0", "1"}:
    raise SystemExit("Unexpected target values.")

if not SMOKE_IDS.issubset(set(ids)):
    raise SystemExit("Some smoke-test IDs are missing from train.csv.")

rng = random.Random(SEED)
dev = []
heldout = []

for target in ("0", "1"):
    class_rows = [r for r in rows if r["target"] == target]

    if len(class_rows) != 750:
        raise SystemExit(
            f"Expected 750 samples for target {target}, "
            f"found {len(class_rows)}."
        )

    forced = [r for r in class_rows if r["id"] in SMOKE_IDS]
    remaining = [r for r in class_rows if r["id"] not in SMOKE_IDS]

    if len(forced) > SPLIT_SIZE:
        raise SystemExit("Too many forced smoke IDs in one class.")

    rng.shuffle(remaining)

    additional_needed = SPLIT_SIZE - len(forced)

    class_dev = forced + remaining[:additional_needed]
    class_heldout = remaining[additional_needed:]

    dev.extend(class_dev)
    heldout.extend(class_heldout)

# Stable output ordering.
dev.sort(key=lambda r: r["id"])
heldout.sort(key=lambda r: r["id"])

dev_ids = {r["id"] for r in dev}
heldout_ids = {r["id"] for r in heldout}

assert len(dev) == 750
assert len(heldout) == 750
assert not dev_ids.intersection(heldout_ids)
assert SMOKE_IDS.issubset(dev_ids)
assert not SMOKE_IDS.intersection(heldout_ids)

OUT.mkdir(parents=True, exist_ok=True)

for filename, dataset in [
    ("dev.csv", dev),
    ("heldout.csv", heldout)
]:
    with (OUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(dataset)

manifest = {
    "source": str(SOURCE),
    "source_sha256": source_hash,
    "seed": SEED,
    "development_count": len(dev),
    "heldout_count": len(heldout),
    "development_targets": dict(Counter(r["target"] for r in dev)),
    "heldout_targets": dict(Counter(r["target"] for r in heldout)),
    "smoke_test_ids": sorted(SMOKE_IDS),
    "smoke_test_count_in_development": len(SMOKE_IDS),
    "development_ids": sorted(dev_ids),
    "heldout_ids": sorted(heldout_ids)
}

with (OUT / "split_manifest.json").open("w") as f:
    json.dump(manifest, f, indent=2)

print("Split created successfully.")
print("Development:", len(dev), Counter(r["target"] for r in dev))
print("Heldout:    ", len(heldout), Counter(r["target"] for r in heldout))
print("Smoke IDs in development:", len(SMOKE_IDS & dev_ids))
print("Overlap:", len(dev_ids & heldout_ids))
print("Output directory:", OUT)
