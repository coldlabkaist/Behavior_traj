from __future__ import annotations

import argparse

import sys

from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.core.mom_pup.repeated_measures import METRIC as INTERNAL_METRIC, _validated_complete_data, split_plot_anova, tukey_emm_pairwise

SOURCE_TIME = "total_stable_proximity_time_sec"

SOURCE_METRIC = "pct_time_stable_proximity"

from analysis.core.mom_pup.stable_proximity_pct_rm_anova import derive_percentage

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Express stable maternal proximity as a percentage of frames with "
            "valid mom-cluster tracking, then run a 2 x 3 mixed repeated-"
            "measures ANOVA and Holm-adjusted within-PND contrasts."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "behavior_metrics"
            / "session_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "output" / "mom_pup" / "stable_proximity_percentage"
        ),
    )
    args = parser.parse_args(argv)

    derived = derive_percentage(pd.read_csv(args.metrics))
    data = _validated_complete_data(
        derived.rename(columns={SOURCE_METRIC: INTERNAL_METRIC})
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anova = split_plot_anova(data)
    emms, all_pairs, same_pnd, model_info = tukey_emm_pairwise(data, anova)
    for frame in (anova, emms, all_pairs, same_pnd, model_info):
        frame.insert(0, "metric", SOURCE_METRIC)

    derived.to_csv(output_dir / "session_metrics_percentage.csv", index=False)
    anova.to_csv(output_dir / "mixed_rm_anova.csv", index=False)
    emms.to_csv(output_dir / "estimated_marginal_means.csv", index=False)
    all_pairs.to_csv(
        output_dir / "all_emm_pairwise_comparisons.csv", index=False
    )
    same_pnd.to_csv(
        output_dir / "mixed_model_simple_effects_group_within_pnd.csv",
        index=False,
    )
    model_info.to_csv(output_dir / "mixed_model_info.csv", index=False)

    print("Stable proximity percentage mixed repeated-measures ANOVA")
    print(anova.to_string(index=False))
    print("\nHolm-adjusted same-PND Control vs VPA simple effects")
    print(
        same_pnd[
            [
                "pnd",
                "estimate_1",
                "estimate_2",
                "difference_1_minus_2",
                "p_raw_two_sided",
                "p_holm_across_3_pnd",
                "significance_holm",
            ]
        ].to_string(index=False)
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
