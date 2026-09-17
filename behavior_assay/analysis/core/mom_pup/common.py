from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, RAW_DATA, resolve_input_path
import re
from pathlib import Path

DEFAULT_RAW_DIR = RAW_DATA["mom_pup"]

DEFAULT_OUTPUT_DIR = ROOT / "output" / "mom_pup"

CONDITION_LABELS = {
	"cont": ("cont", "Control"),
	"control": ("cont", "Control"),
	"exp": ("exp", "VPA"),
	"vpa": ("exp", "VPA"),
}

_SESSION_RE = re.compile(
	r"^(?P<subject>B6_(?:PUP|VPA)_\d+(?:_\d+)?)_"
	r"(?P<condition>cont|control|exp|vpa)_\s*PND_?(?P<pnd>\d+)$",
	flags=re.IGNORECASE,
)

def to_abs_path(path: str | Path) -> Path:
    return resolve_input_path(path)

def canonical_subject_id(value: str) -> str:
	return "_".join(str(value).upper().split("_"))

def parse_session_stem(stem: str) -> dict[str, object]:
	match = _SESSION_RE.fullmatch(stem.strip())
	if not match:
		return {
			"session_id": stem,
			"subject_id": "",
			"condition_code": "",
			"condition": "",
			"pnd": None,
			"metadata_ok": False,
		}
	values = match.groupdict()
	subject_id = canonical_subject_id(values["subject"])
	condition_code, condition = CONDITION_LABELS[str(values["condition"]).lower()]
	pnd = int(values["pnd"])
	return {
		"session_id": f"{subject_id}_{condition_code}_PND{pnd}",
		"subject_id": subject_id,
		"condition_code": condition_code,
		"condition": condition,
		"pnd": pnd,
		"metadata_ok": True,
	}

def safe_filename(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
