from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import pandas as pd
from analysis.core.mom_pup.repeated_measures import METRIC as INTERNAL_METRIC, _validated_complete_data, split_plot_anova, tukey_emm_pairwise

SOURCE_METRIC = "avg_body_scale_proximity_bout_duration_sec"

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2 x 3 mixed repeated-measures ANOVA for mean duration "
            "of body-scale maternal proximity bouts lasting at least one "
            "second."
        )
    )
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity_bouts_1s"
            / "session_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "output"
            / "mom_pup"
            / "body_scale_proximity_bout_duration_1s"
        ),
    )
    args = parser.parse_args(argv)

    raw = pd.read_csv(args.metrics)
    if SOURCE_METRIC not in raw.columns:
        raise ValueError(f"Missing required metric: {SOURCE_METRIC}")
    raw = raw.drop(columns=[INTERNAL_METRIC], errors="ignore")
    data = _validated_complete_data(
        raw.rename(columns={SOURCE_METRIC: INTERNAL_METRIC})
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anova = split_plot_anova(data)
    emms, all_pairs, same_pnd, model_info = tukey_emm_pairwise(data, anova)
    for frame in (anova, emms, all_pairs, same_pnd, model_info):
        frame.insert(0, "metric", SOURCE_METRIC)

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

    print("Mean proximity-bout duration mixed repeated-measures ANOVA")
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

if __name__ == '__main__':
    raise SystemExit(main())
