from __future__ import annotations

import argparse

import sys

import warnings

from pathlib import Path

import numpy as np

import pandas as pd

import patsy

from scipy import stats

from statsmodels.formula.api import mixedlm

from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PNDS = (10, 15, 20)

CONDITIONS = ("Control", "VPA")

METRIC = "pct_time_body_scale_proximity"

from analysis.core.mom_pup.repeated_measures import _stars

from analysis.core.mom_pup.repeated_measures import _validated_complete_data

from analysis.core.mom_pup.repeated_measures import split_plot_anova

from analysis.core.mom_pup.repeated_measures import tukey_emm_pairwise

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2 x 3 mixed repeated-measures ANOVA and Tukey-adjusted "
            "Group x PND estimated-marginal-mean comparisons."
        )
    )
    from analysis.paths import FINAL, REPRODUCED
    parser.add_argument("--metrics", default=str(FINAL / "Fig5F/data/cage_proximity.csv"))
    parser.add_argument("--output-dir", default=str(REPRODUCED / "Fig5F/stat"))
    parser.add_argument("--metric-column", default="pct_time_sustained_body_scale_proximity")
    args = parser.parse_args(argv)

    raw = pd.read_csv(args.metrics)
    if args.metric_column not in raw.columns:
        raise ValueError(f"Missing metric column: {args.metric_column}")
    if args.metric_column != METRIC:
        raw[METRIC] = pd.to_numeric(raw[args.metric_column], errors="raise")
    data = _validated_complete_data(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anova = split_plot_anova(data)
    emms, all_pairs, same_pnd, model_info = tukey_emm_pairwise(data, anova)

    anova.to_csv(output_dir / "mixed_rm_anova.csv", index=False)
    emms.to_csv(output_dir / "estimated_marginal_means.csv", index=False)
    all_pairs.to_csv(output_dir / "tukey_hsd_emm_all_pairs.csv", index=False)
    same_pnd.to_csv(
        output_dir / "tukey_hsd_emm_same_pnd_control_vs_vpa.csv",
        index=False,
    )
    same_pnd.to_csv(
        output_dir / "mixed_model_simple_effects_group_within_pnd.csv",
        index=False,
    )
    model_info.to_csv(output_dir / "mixed_model_info.csv", index=False)

    print(f"Mixed repeated-measures ANOVA: {args.metric_column}")
    print(anova.to_string(index=False))
    print("\nTukey-adjusted same-PND Control vs VPA comparisons")
    print(
        same_pnd[
            [
                "pnd",
                "estimate_1",
                "estimate_2",
                "difference_1_minus_2",
                "p_tukey_hsd",
                "p_holm_across_3_pnd",
                "significance_holm",
                "significance",
            ]
        ].to_string(index=False)
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
