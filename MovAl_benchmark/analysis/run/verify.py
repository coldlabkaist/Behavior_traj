"""Compare migrated results with preserved values; report unresolved differences."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from ..core.paths import ROOT, JITTER_PANEL
from ..core.jitter import plot_summary


def compare_tables(actual, expected, keys):
    a, b = pd.read_csv(actual), pd.read_csv(expected)
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False,
                                  check_exact=False, rtol=1e-9, atol=1e-9)
    return len(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/reproduced")
    args = parser.parse_args()
    out = args.output_root
    checks = []

    def check(name, operation):
        try:
            detail = operation()
            checks.append(dict(check=name, passed=True, detail=str(detail)))
        except (AssertionError, ValueError, KeyError, OSError) as exc:
            checks.append(dict(check=name, passed=False, detail=str(exc)[:500]))

    tables = [
        ("Fig2B", "data/video_level_tracking_miss.csv", ["video", "keypoint", "input", "method"]),
        ("Fig2B", "stat/video_level_descriptive_stats.csv", ["keypoint", "condition"]),
        ("Fig2B", "stat/two_way_repeated_measures_anova_by_keypoint.csv", ["keypoint", "effect"]),
        ("Fig2B", "stat/paired_posthoc_holm_all_15_pairs_by_keypoint.csv", ["keypoint", "condition_1", "condition_2"]),
        ("Fig2C", "data/identity_switch_frequencies.csv", ["videos"]),
        ("Fig2C", "stat/repeated_measures_anova.csv", ["effect"]),
        ("Fig2C", "stat/paired_ttests_bh.csv", ["test_column"]),
        (JITTER_PANEL, "data/fft_band_power_by_video_track.csv", ["video", "track", "keypoint", "band", "method"]),
        (JITTER_PANEL, "data/fft_band_power_by_video.csv", ["video", "keypoint", "band", "method"]),
        (JITTER_PANEL, "stat/fft_band_power_summary_video_level.csv", ["keypoint", "band", "method"]),
        (JITTER_PANEL, "stat/two_way_repeated_measures_anova.csv", ["keypoint", "band", "effect"]),
        (JITTER_PANEL, "stat/paired_posthoc_holm.csv", ["keypoint", "band", "group1", "group2"]),
    ]
    for panel, file, keys in tables:
        check(f"{panel}/{file}", lambda p=panel, f=file, k=keys: compare_tables(out / p / f, ROOT / p / f, k))

    for panel, filename in [("Fig2B", "tracking_miss.png"), ("Fig2C", "identity_switch_frequency.png"), ("Fig2E", "rmse_heatmap.png")]:
        def pixels(p=panel, f=filename):
            with Image.open(ROOT / p / "figure" / f) as x, Image.open(out / p / "figure" / f) as y:
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            return "pixel-identical PNG"
        check(f"{panel}/figure", pixels)

    actual = pd.read_csv(out / "Fig2B/data/plot_summary.csv")
    expected = pd.read_csv(ROOT / "Fig2B/data/plot_summary.csv")
    merged = expected.merge(actual, on=["keypoint", "method", "input"], suffixes=("_saved", "_current"), validate="one_to_one")
    merged["matches"] = np.isclose(merged.mean_saved, merged.mean_current, atol=1e-9, rtol=0) & np.isclose(merged.sem_saved, merged.sem_current, atol=1e-9, rtol=0) & merged.n_current.eq(17) & merged.n_saved.eq(17)
    merged.to_csv(out / "missing_plot_comparison.csv", index=False)
    checks.append(dict(check="Fig2B/source_vs_adopted_plot", passed=bool(merged.matches.all()),
                       detail=f"{merged.matches.sum()}/30 mean and SEM cells match adopted 17-video summary"))

    expected = pd.read_csv(ROOT / JITTER_PANEL / "data/saved_plot_checks.csv")
    actual = plot_summary(pd.read_csv(out / JITTER_PANEL / "data/fft_band_power_by_video_track.csv"))
    merged = expected.merge(actual, on=["keypoint", "band", "method"], validate="one_to_one")
    check("jitter/saved_plot_scalar_checks", lambda: np.testing.assert_allclose(
        merged.mean_plus_sem, merged.mean_power + merged.se_power, rtol=1e-12, atol=1e-9))

    expected = pd.read_csv(ROOT / "Fig2E/data/rmse_summary.csv")
    actual = pd.read_csv(out / "Fig2E/data/rmse_from_coordinates.csv")
    merged = expected.merge(actual, on=["keypoint", "method", "input"], suffixes=("_saved", "_current"), validate="one_to_one")
    merged["matches_full_precision"] = np.isclose(merged.mse_saved, merged.mse_current, atol=1e-12, rtol=0) & np.isclose(merged.rmse_saved, merged.rmse_current, atol=1e-12, rtol=0) & merged.n_pairs_saved.eq(merged.n_pairs_current)
    merged.to_csv(out / "rmse_coordinate_comparison.csv", index=False)
    checks.append(dict(check="Fig2E/coordinates_vs_adopted_RMSE", passed=bool(merged.matches_full_precision.all()),
                       detail=f"{merged.matches_full_precision.sum()}/30 MSE, RMSE and valid-pair counts match"))
    report = pd.DataFrame(checks)
    report.to_csv(out / "verification.csv", index=False)
    print(report.to_string(index=False))
    if not report.passed.all():
        print("Reproduction differs from adopted results; inspect verification.csv before replacing outputs.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
