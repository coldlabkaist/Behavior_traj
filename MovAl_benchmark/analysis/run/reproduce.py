"""Run from the repository root: python -m MovAl_benchmark.analysis.run.reproduce.

Default outputs stay in work/reproduced for comparison with adopted results.
Existing adopted figures and statistics are never overwritten by default.
"""
import argparse
from pathlib import Path
import pandas as pd
from ..core import missing, identity, jitter, rmse
from ..core.paths import ROOT, JITTER_PANEL
from ..plots import benchmark, jitter as jitter_plots
from ..plots.style import apply_style, save_figure, significance_label


def write_csv(data, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / name, index=False)


def reproduce_missing(destination):
    out = destination / "Fig2B"
    data = missing.load_video_level_data()
    write_csv(data, out / "data", "video_level_tracking_miss.csv")
    write_csv(missing.descriptive_stats(data), out / "stat", "video_level_descriptive_stats.csv")
    write_csv(missing.repeated_anova(data), out / "stat", "two_way_repeated_measures_anova_by_keypoint.csv")
    write_csv(missing.holm_posthoc(data), out / "stat", "paired_posthoc_holm_all_15_pairs_by_keypoint.csv")
    summary = missing.plot_summary(pd.read_csv(missing.DATA_PATH))
    write_csv(summary, out / "data", "plot_summary.csv")
    save_figure(benchmark.plot_missing(summary), out / "figure", "tracking_miss")


def reproduce_identity(destination):
    out = destination / "Fig2C"
    data = identity.load_data()
    anova, pairs = identity.statistics(data)
    pairs["significance"] = pairs.q.map(significance_label)
    write_csv(data, out / "data", "identity_switch_frequencies.csv")
    write_csv(anova, out / "stat", "repeated_measures_anova.csv")
    write_csv(pairs, out / "stat", "paired_ttests_bh.csv")
    (out / "stat/caption.txt").write_text(identity.caption(anova, pairs), encoding="utf-8")
    save_figure(benchmark.plot_identity(data, pairs), out / "figure", "identity_switch_frequency")


def reproduce_jitter(destination):
    out = destination / JITTER_PANEL
    track = jitter.compute_band_power()
    video = jitter.video_summary(track)
    anova, pairs = jitter.repeated_measures_statistics(video)
    write_csv(track, out / "data", "fft_band_power_by_video_track.csv")
    write_csv(video, out / "data", "fft_band_power_by_video.csv")
    write_csv(jitter.descriptive_summary(video), out / "stat", "fft_band_power_summary_video_level.csv")
    write_csv(anova, out / "stat", "two_way_repeated_measures_anova.csv")
    write_csv(pairs, out / "stat", "paired_posthoc_holm.csv")
    traces = jitter.representative_traces()
    summary = jitter.plot_summary(track)
    for keypoint, name in [("Body_C", "body_center"), ("Nose", "nose"), ("Tail", "tail")]:
        save_figure(jitter_plots.plot_bands(summary, keypoint, pairs), out / "figure", f"{name}_fft")
        save_figure(jitter_plots.plot_traces(traces, keypoint), out / "figure", f"{name}_movement")


def reproduce_rmse(destination):
    out = destination / "Fig2E"
    calculated = rmse.calculate_from_coordinates()
    write_csv(calculated, out / "data", "rmse_from_coordinates.csv")
    write_csv(calculated, out / "data", "rmse_summary.csv")
    save_figure(benchmark.plot_rmse(calculated), out / "figure", "rmse_heatmap", transparent=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "Fig2B", "Fig2C", "Fig2D", "FigS1", "Fig2E"], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/reproduced")
    args = parser.parse_args()
    apply_style()
    jobs = {"Fig2B": reproduce_missing, "Fig2C": reproduce_identity,
            "Fig2D": reproduce_jitter, "Fig2E": reproduce_rmse}
    selection = list(jobs) if args.figure == "all" else ["Fig2D" if args.figure == "FigS1" else args.figure]
    for panel in selection:
        jobs[panel](args.output_root)
        print(f"{panel}: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
