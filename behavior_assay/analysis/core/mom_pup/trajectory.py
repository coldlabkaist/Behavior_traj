from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import re

TRACK_LABELS = {
	"track_0": "Mom",
	"track_1": "Pup_1",
	"track_2": "Pup_2",
	"track_3": "Pup_3",
	"track_4": "Pup_4",
	"track_5": "Pup_5",
	"track_6": "Pup_6",
}

TRACK_COLORS = {
	"track_0": (128/255, 0/255, 0/255),        # \x80\x00\x00
	"track_1": (0/255, 128/255, 0/255),        # \x00\x80\x00
	"track_2": (128/255, 128/255, 0/255),      # \x80\x80\x00
	"track_3": (0/255, 0/255, 128/255),        # \x00\x00\x80
	"track_4": (128/255, 0/255, 128/255),      # \x80\x00\x80
	"track_5": (0/255, 128/255, 128/255),      # \x00\x80\x80
	"track_6": (255/255, 140/255, 0/255),      # \xff\x8c\x00
}

TRACK_COLORS_FIXED = {
	"track_0": "#d62728",  # red
	"track_1": "#2ca02c",  # green
	"track_2": "#ffbf00",  # yellow/amber
	"track_3": "#1f77b4",  # blue
}

def load_csvs(path: Path) -> list[Path]:
	if path.is_file():
		return [path]
	return sorted([p for p in path.glob("*.csv")])

def infer_title_from_name(name: str) -> str:
	# Try to parse PND and PUP id for a concise title: "PND {pnd} cage {pup:02d}"
	pnd = None
	pup = None
	m = re.search(r"PND\s*(\d+)", name, re.IGNORECASE)
	if not m:
		m = re.search(r"PND(\d+)", name, re.IGNORECASE)
	if m:
		pnd = m.group(1)
	m2 = re.search(r"PUP[_-]?(\d+)", name, re.IGNORECASE)
	if m2:
		pup = m2.group(1)
	if pnd and pup:
		return f"PND {pnd} cage {int(pup):02d}"
	if pnd:
		return f"PND {pnd}"
	return name

def _generic_track_label(track: str) -> str:
	m = re.match(r"track_(\d+)$", str(track))
	if m:
		return f"Track {int(m.group(1))}"
	return str(track)

def _build_track_labels(tracks: list[str], label_mode: str) -> dict[str, str]:
	if label_mode == "mom_pup":
		return {t: TRACK_LABELS.get(t, t) for t in tracks}
	return {t: _generic_track_label(t) for t in tracks}
