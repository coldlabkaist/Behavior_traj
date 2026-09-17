from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, relative_input_path
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

PHASE_RE = re.compile(r"(?:(?<=_)|^)(hab|soc|nov)(?=(?:_|$))", re.IGNORECASE)

TRIAL_RE = re.compile(r"(?:(?<=_)|^)id(\d+)(?=(?:_|$))", re.IGNORECASE)

SESSION_RE = re.compile(r"(?:(?<=_)|^)n_(\d+)(?=(?:_|$))", re.IGNORECASE)

SEX_RE = re.compile(r"(?:(?<=_)|^)(m|f)(?=(?:_|$))", re.IGNORECASE)

LAYOUT_TOKEN_RE = re.compile(r"(?:(?<=_)|^)(s[12]_[lr])(?=(?:_|$))", re.IGNORECASE)

DATE_DIR_RE = re.compile(r"^\d{8}$")

HAB_TYPE_RE = re.compile(r"(?:(?<=_)|^)(open|closed)(?=(?:_|$))", re.IGNORECASE)

def _normalize_stem(stem: str) -> str:
	cleaned = stem.strip()
	cleaned = re.sub(r"(?i)\.csv$", "", cleaned)
	cleaned = re.sub(r"\s+", "", cleaned)
	cleaned = re.sub(r"_+", "_", cleaned).strip("_")
	cleaned = re.sub(r"(?i)^predict__?", "", cleaned)
	cleaned = re.sub(r"(?i)_pins$", "", cleaned)
	cleaned = re.sub(r"(?i)\.(mp4|avi|mov|mkv)$", "", cleaned)
	cleaned = re.sub(r"(?i)(?:^|_)3chamber(?=_|$)", "_", cleaned)
	cleaned = re.sub(r"(?i)(?:^|_)3c(?=_|$)", "_", cleaned)
	cleaned = re.sub(r"_(?:\d{6}|\d{8})_\d{6}$", "", cleaned)
	cleaned = re.sub(r"_+", "_", cleaned).strip("_")
	return cleaned

def _infer_context(path: Path, input_root: Path) -> tuple[str, str]:
	rel_parts = path.relative_to(input_root).parts
	date = ""
	date_idx = -1
	for idx, part in enumerate(rel_parts[:-1]):
		if DATE_DIR_RE.fullmatch(str(part)):
			date = str(part)
			date_idx = idx
	if not date:
		raise ValueError(f"Could not infer date folder for path: {path}")

	condition = ""
	if date_idx > 0:
		candidate = str(rel_parts[date_idx - 1]).strip()
		if candidate and candidate.lower() != "object":
			condition = candidate
	return condition, date

def _session_key(path: Path, input_root: Path) -> tuple[str, str, str]:
	condition, date = _infer_context(path, input_root)
	return condition, date, _normalize_stem(path.stem).lower()

def _format_layout_token(token: str) -> str:
	token_upper = token.upper()
	if token_upper.startswith("S1_"):
		return f"S1_{token_upper[-1].lower()}"
	if token_upper.startswith("S2_"):
		return f"S2_{token_upper[-1].lower()}"
	return token

def _path_for_csv(path: Path) -> str:
    return relative_input_path(path)

def _opposite_side(side: str) -> str:
	if side == "l":
		return "r"
	if side == "r":
		return "l"
	return ""

def _parse_condition_map(arg: str) -> dict[str, str]:
	if not str(arg).strip():
		return {}
	parsed: dict[str, str] = {}
	for chunk in str(arg).split(","):
		item = chunk.strip()
		if not item:
			continue
		if "=" not in item:
			raise ValueError(f"Invalid --condition-map item: {item!r}. Use DATE=LABEL.")
		date_str, label = item.split("=", 1)
		date_str = date_str.strip()
		label = label.strip()
		if not date_str or not label:
			raise ValueError(f"Invalid --condition-map item: {item!r}. Use DATE=LABEL.")
		parsed[date_str] = label
	return parsed

def _parse_metadata(path: Path) -> dict[str, object]:
	cleaned = _normalize_stem(path.stem)
	cleaned_lower = cleaned.lower()
	phase_match = PHASE_RE.search(cleaned_lower)
	subject_id = cleaned[: phase_match.start()].rstrip("_") if phase_match else ""
	phase = phase_match.group(1).lower() if phase_match else ""
	hab_type_match = HAB_TYPE_RE.search(cleaned_lower) if phase == "hab" else None
	hab_type = hab_type_match.group(1).lower() if hab_type_match else ("closed" if phase == "hab" else "")
	phase_detail = f"hab_{hab_type}" if phase == "hab" and hab_type else phase
	trial_match = TRIAL_RE.search(cleaned_lower)
	session_match = SESSION_RE.search(cleaned_lower)
	sex_match = SEX_RE.search(subject_id.lower())
	layout_tokens = [_format_layout_token(token) for token in LAYOUT_TOKEN_RE.findall(cleaned_lower)]
	session_n = int(session_match.group(1)) if session_match else pd.NA
	if phase in {"soc", "nov"} and pd.isna(session_n):
		session_n = 1

	return {
		"canonical_stem": cleaned_lower,
		"session_stem": cleaned,
		"subject_id": subject_id,
		"sex": sex_match.group(1).lower() if sex_match else "",
		"phase": phase,
		"phase_detail": phase_detail,
		"hab_type": hab_type,
		"trial_id": (int(trial_match.group(1)) if trial_match else pd.NA),
		"session_n": session_n,
		"s1_side": next((token[-1] for token in layout_tokens if token.startswith("S1_")), ""),
		"s2_side": next((token[-1] for token in layout_tokens if token.startswith("S2_")), ""),
		"layout_raw": "_".join(layout_tokens),
	}

def _collect_files(
	input_root: Path,
	*,
	excluded_dirs: set[str] | None = None,
) -> tuple[dict[tuple[str, str, str], list[Path]], dict[tuple[str, str, str], list[Path]]]:
	pose_map: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
	pins_map: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
	excluded = {name.lower() for name in (excluded_dirs or set())}

	for csv_path in sorted(input_root.rglob("*.csv")):
		if ".before_repin_" in csv_path.name.lower():
			continue
		relative_parts = {part.lower() for part in csv_path.relative_to(input_root).parts[:-1]}
		if relative_parts & excluded:
			continue
		try:
			key = _session_key(csv_path, input_root)
		except ValueError:
			print(f"Skipped CSV without condition/date context: {csv_path.relative_to(input_root).as_posix()}")
			continue
		if csv_path.parent.name.lower() == "object":
			pins_map[key].append(csv_path)
		else:
			pose_map[key].append(csv_path)

	return pose_map, pins_map

def _build_rows(
	pose_map: dict[tuple[str, str, str], list[Path]],
	pins_map: dict[tuple[str, str, str], list[Path]],
	*,
	condition_map: dict[str, str],
) -> list[dict[str, object]]:
	rows: list[dict[str, object]] = []

	for inferred_condition, date, canonical_stem in sorted(set(pose_map) | set(pins_map)):
		pose_paths = pose_map.get((inferred_condition, date, canonical_stem), [])
		pins_paths = pins_map.get((inferred_condition, date, canonical_stem), [])
		reference_path = pose_paths[0] if pose_paths else pins_paths[0]
		meta = _parse_metadata(reference_path)
		condition = inferred_condition or condition_map.get(str(date), "")
		social_side = meta["s1_side"] if meta["phase"] == "soc" else ""
		empty_side = _opposite_side(str(social_side)) if social_side else ""
		familiar_side = meta["s1_side"] if meta["phase"] == "nov" else ""
		novel_side = meta["s2_side"] if meta["phase"] == "nov" else ""
		heatmap_target_side = social_side if meta["phase"] == "soc" else novel_side if meta["phase"] == "nov" else ""
		heatmap_target_label = "social" if meta["phase"] == "soc" else "novel" if meta["phase"] == "nov" else ""
		mirror_lr_for_target_right = bool(heatmap_target_side == "l")

		issues: list[str] = []
		if not pose_paths:
			issues.append("missing_pose")
		if not pins_paths:
			issues.append("missing_pins")
		if len(pose_paths) > 1:
			issues.append("duplicate_pose")
		if len(pins_paths) > 1:
			issues.append("duplicate_pins")
		if not meta["phase"]:
			issues.append("phase_parse_failed")
		if not meta["subject_id"]:
			issues.append("subject_parse_failed")
		if pd.isna(meta["trial_id"]):
			issues.append("missing_trial_id")
		if meta["phase"] in {"soc", "nov"} and pd.isna(meta["session_n"]):
			issues.append("missing_session_n")
		if meta["phase"] == "hab" and meta["layout_raw"]:
			issues.append("unexpected_layout_in_hab")
		if meta["phase"] == "soc":
			if not meta["s1_side"]:
				issues.append("missing_s1_side_in_soc")
			if meta["s2_side"]:
				issues.append("unexpected_s2_side_in_soc")
		if meta["phase"] == "nov":
			if not meta["s1_side"] or not meta["s2_side"]:
				issues.append("incomplete_layout_in_nov")
		if not condition:
			issues.append("missing_condition")

		rows.append(
			{
				"date": date,
				"condition": condition,
				"session_key": f"{(condition or 'unknown').lower()}__{date}__{canonical_stem}",
				**meta,
				"social_side": social_side,
				"empty_side": empty_side,
				"familiar_side": familiar_side,
				"novel_side": novel_side,
				"heatmap_target_label": heatmap_target_label,
				"heatmap_target_side": heatmap_target_side,
				"mirror_lr_for_target_right": mirror_lr_for_target_right,
				"pose_count": len(pose_paths),
				"pins_count": len(pins_paths),
				"pose_path": ";".join(_path_for_csv(path) for path in pose_paths),
				"pins_path": ";".join(_path_for_csv(path) for path in pins_paths),
				"is_complete": bool(pose_paths and pins_paths and len(pose_paths) == 1 and len(pins_paths) == 1),
				"issues": ";".join(issues),
			}
		)

	return rows
