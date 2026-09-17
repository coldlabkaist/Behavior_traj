"""Shared feature metadata and the frozen four-motif column schema.

Cache indices, tuple order and column names are part of the published data
contract. BOI's median/IQR features remain defined in boi.py; they are a
different feature set from the relation features described here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    cache_index: int
    key: str
    label: str
    family: str
    inverse: str
    display_unit: str


# Biological relation features stored in the DAE cache.
FEATURES = (
    Feature(0, "body_distance", "Body distance", "Spacing", "expm1", "body lengths"),
    Feature(1, "nose_nose_distance", "Nose–Nose distance", "Spacing", "expm1", "body lengths"),
    Feature(2, "nose_tailbase_distance", "Nose–Tailbase distance", "Spacing", "expm1", "body lengths"),
    Feature(3, "mutual_facing", "Mutual facing", "Orientation", "identity", "cosine score"),
    Feature(4, "approach_rate", "Approach rate", "Orientation", "sinh", "body lengths/s"),
    Feature(5, "relative_speed", "Relative speed", "Dynamics", "expm1", "body lengths/s"),
    Feature(7, "dyadic_grouping", "Dyadic grouping", "Spacing", "identity", "index"),
    Feature(8, "configuration_speed", "Configuration speed", "Dynamics", "expm1", "body lengths/s"),
)

# Ordered subset used to calculate FigS5A motif novelty.
PROFILE_KEYS = (
    "body_distance",
    "mutual_facing",
    "approach_rate",
    "relative_speed",
    "dyadic_grouping",
    "configuration_speed",
)

# Component order and labels shared by the motif calculations and figures.
TOKENS = ("M0", "M1", "M2", "M3")
MOTIF_NAMES = (
    "Compact-static",
    "Dispersal / non-oriented dynamic",
    "Separated-static",
    "Approach-oriented dynamic",
)
OCCUPANCY_COLUMNS = tuple(f"p_{token}" for token in TOKENS)
TRANSITION_COLUMNS = tuple(
    f"{source}_to_{target}" for source in TOKENS for target in TOKENS
)
