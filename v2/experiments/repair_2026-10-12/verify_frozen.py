#!/usr/bin/env python3
"""Check the shipped repair-round files against the SHA-256 digests that the registrations froze.

Run from anywhere:  python3 verify_frozen.py [--root <dir that holds paper_a, round3 ... round7>]

For every RECIPE*_FROZEN.json it recomputes the digest of each listed code file and of the features
source, and for every frozen_*.sha256 / PREREG_n_SHA256 it recomputes the digest of the file it seals.
The per-cell forecast and model files are not shipped (see the README), so the R*_VAL/TEST_SHA256 lists
are not re-checked here. Exit status 0 only if nothing mismatches.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HEX = re.compile(r"^[0-9a-f]{64}$")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
root = Path(ap.parse_args().root).resolve()
exp = root.parent  # .../experiments, where ../fix_2026-10-11 lives
bad, ok, skipped = [], 0, []


# release v2.0.0 replaced two internal words in the docstring of fix_2026-10-11/monitor_far.py; undoing exactly those two
# edits must give back the file that the registrations of rounds 6 and 7 froze.
V200_EDITS = [
    ('"""audit item 3:', '"""VA7 item 3:'),
    (
        "Algorithm 3 in an earlier draft describes",
        "Algorithm 3 in the thesis describes",
    ),
]


def digest(c: Path) -> str:
    if c.name == "monitor_far.py":
        t = c.read_text()
        for new, old in V200_EDITS:
            t = t.replace(new, old, 1)
        return hashlib.sha256(t.encode()).hexdigest()
    return sha(c)


def check(label: str, cands, want: str):
    global ok
    for c in cands:
        if c.exists():
            if digest(c) == want:
                ok += 1
            else:
                bad.append(f"MISMATCH {label}: {c}")
            return
    skipped.append(f"not shipped {label}: {cands[0].name}")


for rec in sorted(root.glob("round*/RECIPE*FROZEN.json")):
    d = json.loads(rec.read_text())
    here = rec.parent
    fs = d.get("features_source") or {}
    if fs.get("file") and HEX.match(fs.get("sha256", "")):
        check(
            f"{rec.name} features_source",
            [
                root / Path(fs["file"]).relative_to("experiments/repair_2026-10-12"),
                here / Path(fs["file"]).name,
            ],
            fs["sha256"],
        )
    for key, val in d.items():
        if (
            isinstance(val, dict)
            and val
            and all(isinstance(x, str) and HEX.match(x) for x in val.values())
        ):
            for rel, want in val.items():
                check(
                    f"{rec.name}:{key}",
                    [
                        here / rel,
                        root / rel,
                        (root / rel).resolve(),
                        exp / rel.lstrip("./"),
                    ]
                    + [
                        root / r / rel
                        for r in ("paper_a", "round3", "round4", "round5")
                    ],
                    want,
                )

for s in sorted(root.glob("*/frozen_*.sha256")):
    j = s.with_suffix(".json")
    if j.exists():
        check(s.name, [j], s.read_text().split()[0])
for s in sorted(root.glob("round*/PREREG_*_SHA256")):
    n = re.search(r"PREREG_(\d+)_SHA256", s.name).group(1)
    check(s.name, [s.parent / f"REPAIR_PREREG_{n}.md"], s.read_text().split()[0])

print(
    f"{ok} digests match, {len(bad)} mismatch, {len(skipped)} listed files not shipped"
)
for line in bad + skipped:
    print(" ", line)
sys.exit(1 if bad else 0)
