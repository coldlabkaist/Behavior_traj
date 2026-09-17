"""Compare regenerated paper results with the curated final archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify(output_dir: Path) -> None:
    final = PROJECT_ROOT / "output/final"
    reference = pd.read_csv(final / "Fig5D/data/individual_behavior_values.csv")
    actual = pd.read_csv(output_dir / "individual_behavior_values.csv")
    pd.testing.assert_frame_equal(actual, reference, rtol=1e-12, atol=1e-12)
    stats = pd.read_csv(output_dir / "week_stats.csv")
    expected_stats = pd.concat([
        pd.read_csv(final / "Fig5D/stat/Fig5D_statistics.csv"),
        pd.read_csv(final / "FigS9AB/stat/FigS9AB_statistics.csv"),
    ], ignore_index=True)
    pd.testing.assert_frame_equal(stats, expected_stats, rtol=1e-12, atol=1e-12)
    categories = {"Fig5D": "social", "FigS9A": "attentive", "FigS9B": "prosocial"}
    for panel in ["Fig5D", "FigS9AB"]:
        manifest = json.loads((final / panel / "manifest.json").read_text(encoding="utf-8"))
        for asset in manifest["files"]:
            preserved = final / panel / asset["file"]
            if hashlib.sha256(preserved.read_bytes()).hexdigest() != asset["sha256"]:
                raise AssertionError(f"Curated file has changed: {preserved}")
            if preserved.suffix != ".png":
                continue
            category = categories[preserved.stem.split("_")[0]]
            generated = output_dir / f"direct_interaction_{category}_all_individual_trajectory_visible.png"
            if hashlib.sha256(generated.read_bytes()).hexdigest() != asset["sha256"]:
                raise AssertionError(f"PNG differs: {generated}; check the rendering environment in README.md")
    print(f"PASS: {len(actual)} observations, {len(stats)} statistical comparisons, 3 identical PNGs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "work/direct_interaction")
    args = parser.parse_args()
    verify(args.output_dir)


if __name__ == "__main__":
    main()
