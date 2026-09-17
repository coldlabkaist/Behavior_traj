from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, RAW_DATA, resolve_input_path
import re
from pathlib import Path

DEFAULT_RAW_DIR = RAW_DATA["oft"]

DEFAULT_OUTPUT_DIR = ROOT / "output" / "OFT"

FPS = 29.0

ARENA_CM = 40.0

CENTRE_FRACTION = 0.50

CONDITION_LABELS = {
	"cont": "Control",
	"control": "Control",
	"exp": "VPA",
	"vpa": "VPA",
}

CAGE_CONDITIONS = {
	"B6_VPA_10": "Control",
	"B6_VPA_10_02": "Control",
	"B6_VPA_11": "VPA",
	"B6_VPA_13": "VPA",
	"B6_VPA_15": "VPA",
	"B6_VPA_17": "VPA",
}

_OFT_RE = re.compile(
	r"^(?P<cage>B6_VPA_\d+(?:_\d+)?)_"
	r"(?P<sex>[fm])_(?P<cohort>\d+)_oft_id(?P<animal_id>\d+)"
	r"(?:_(?P<filename_condition>cont|control|exp|vpa))?$",
	flags=re.IGNORECASE,
)

def to_abs_path(path: str | Path) -> Path:
    return resolve_input_path(path)

def canonical_cage_id(value: str) -> str:
	tokens = str(value).upper().split("_")
	return "_".join(tokens)

def parse_oft_stem(stem: str) -> dict[str, object]:
	match = _OFT_RE.fullmatch(stem.strip())
	if not match:
		return {
			"session_id": stem,
			"subject_id": stem,
			"cage_id": "",
			"sex": "",
			"cohort": "",
			"animal_id": "",
			"condition": "",
			"filename_condition": "",
			"condition_source": "",
			"metadata_ok": False,
		}

	values = match.groupdict()
	cage_id = canonical_cage_id(values["cage"])
	filename_condition = CONDITION_LABELS.get(
		str(values.get("filename_condition") or "").lower(),
		"",
	)
	condition = CAGE_CONDITIONS.get(cage_id, filename_condition)
	session_id = (
		f"{cage_id}_{str(values['sex']).lower()}_{values['cohort']}"
		f"_oft_id{values['animal_id']}"
	)
	return {
		"session_id": session_id,
		"subject_id": session_id,
		"cage_id": cage_id,
		"sex": str(values["sex"]).lower(),
		"cohort": str(values["cohort"]),
		"animal_id": int(values["animal_id"]),
		"condition": condition,
		"filename_condition": filename_condition,
		"condition_source": "cage_override" if cage_id in CAGE_CONDITIONS else "filename",
		"metadata_ok": bool(condition),
	}

def safe_filename(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
