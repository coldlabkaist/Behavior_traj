"""Reproduce Fig5D and FigS9AB from A/B predictions or the curated mouse table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from core.behavior import build_individual_behavior_table
from core.statistics import compute_week_stats, write_coverage
from plots.trajectory import plot_category

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "models": ["modelA", "modelB"],
    "source_schema": "b6_vpa_pnd_id",
    "threshold": 0.5,
    "fps": 30,
    "max_gap_frames": 15,
    "min_bout_frames": 15,
    "frame_policy": "max_zero_pad",
    "categories": ["Social", "Attentive", "Prosocial"],
    "p_mode": "figure-bh",
    "dpi": 300,
    "width_inches": 9.4,
    "height_inches": 4.2,
    "mean_focus": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=PROJECT_ROOT.parent.parent/"data/csv/Fig5D_FigS9AB/predictions")
    parser.add_argument("--input-csv", type=Path,
                        help="Use a curated individual table instead of recomputing A/B predictions")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT/"work/direct_interaction")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (PROJECT_ROOT/"work").resolve() not in output.parents:
        raise ValueError("Output must be a subdirectory of analysis/work")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory to preserve previous results")
    if args.input_csv:
        data = pd.read_csv(args.input_csv)
        for name in ["fps", "max_gap_frames", "min_bout_frames"]:
            if not data[name].eq(SETTINGS[name]).all():
                raise ValueError(f"Input table has different {name}")
        if not data.source_schema.eq(SETTINGS["source_schema"]).all():
            raise ValueError("Input table must contain the B6 analysis cohort")
        qc = None
    else:
        print("Recomputing B6 mouse observations from A/B probabilities", flush=True)
        data, qc = build_individual_behavior_table(
            args.result_dir, threshold=SETTINGS["threshold"], fps=SETTINGS["fps"],
            max_gap_frames=SETTINGS["max_gap_frames"], min_bout_frames=SETTINGS["min_bout_frames"],
        )
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output/"individual_behavior_values.csv", index=False)
    if qc is not None:
        qc.to_csv(output/"frame_alignment.csv", index=False)
    all_stats = []
    for category in SETTINGS["categories"]:
        statistics = compute_week_stats(data, category, SETTINGS["p_mode"])
        all_stats.append(statistics)
        plot_category(data, category, statistics, output, sex_group="all",
                      dpi=SETTINGS["dpi"], width=SETTINGS["width_inches"],
                      height=SETTINGS["height_inches"], mean_focus=SETTINGS["mean_focus"])
    pd.concat(all_stats, ignore_index=True).to_csv(output/"week_stats.csv", index=False)
    write_coverage(data, output)
    (output/"settings.json").write_text(json.dumps(SETTINGS, indent=2)+"\n", encoding="utf-8")
    print(f"Created Fig5D / FigS9AB: {len(data)} observations, {data.mouse_id.nunique()} mice; {output}")


if __name__ == "__main__":
    main()
