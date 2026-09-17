from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MODELS = ("modelA", "modelB")


@dataclass(frozen=True)
class ParsedMetadata:
    file_name: str
    source_schema: str
    cage_number: str
    sex: str
    week: int
    condition: str
    id_number: str
    pnd: int | None = None
    date: str | None = None
    partner_id: str | None = None


@dataclass(frozen=True)
class SampleMetadata:
    sample_key: str
    representative_file_name: str
    source_schema: str
    cage_number: str
    sex: str
    week: int
    condition: str
    id_number: str
    pnd: int | None
    date: str | None
    partner_id: str | None
    modelA_file_name: str
    modelB_file_name: str
    exact_file_name_match: bool


VPA_PATTERN = re.compile(
    r"^(?P<cage>B6_VPA_(?:\d+_)?\d+_[mf]_\d+)_"
    r"PND_?(?P<pnd>\d+)_(?P<condition>cont|exp)_int_id(?P<id_number>\d+)\.csv$"
)

SHORT_B6_PATTERN = re.compile(
    r"^B6_(?P<cage>(?:\d+_)?\d+_[mf]_\d+)_"
    r"PND_?(?P<pnd>\d+)_id(?P<id_number>\d+)_int"
    r"(?:_(?P<condition>cont|exp))?\.csv$"
)

DATED_WEEK_PATTERN = re.compile(
    r"^(?P<cage>(?P<date>\d{6})_(?P<week>\d+)w_"
    r"(?P<id_number>\d+b)_(?P<partner_id>\d+b))_"
    r"(?P<sex>[mf])_int_(?P<condition>cont|exp)\.csv$"
)


def parse_filename(file_name: str) -> ParsedMetadata:
    """Parse a SIMBA prediction CSV filename into sample-level metadata."""
    name = Path(file_name).name

    match = VPA_PATTERN.match(name)
    if match:
        cage_number = match.group("cage")
        pnd = int(match.group("pnd"))
        return ParsedMetadata(
            file_name=name,
            source_schema="b6_vpa_pnd_id",
            cage_number=cage_number,
            sex=_sex_from_cage(cage_number),
            week=pnd // 7,
            condition=match.group("condition"),
            id_number=match.group("id_number"),
            pnd=pnd,
        )

    match = SHORT_B6_PATTERN.match(name)
    if match:
        cage_number = f"B6_VPA_{match.group('cage')}"
        pnd = int(match.group("pnd"))
        return ParsedMetadata(
            file_name=name,
            source_schema="b6_vpa_pnd_id",
            cage_number=cage_number,
            sex=_sex_from_cage(cage_number),
            week=pnd // 7,
            condition=match.group("condition") or "exp",
            id_number=match.group("id_number"),
            pnd=pnd,
        )

    match = DATED_WEEK_PATTERN.match(name)
    if match:
        return ParsedMetadata(
            file_name=name,
            source_schema="dated_week_pair",
            cage_number=match.group("cage"),
            sex=match.group("sex"),
            week=int(match.group("week")),
            condition=match.group("condition"),
            id_number=match.group("id_number"),
            date=match.group("date"),
            partner_id=match.group("partner_id"),
        )

    raise ValueError(f"Unrecognized filename schema: {name}")


def build_metadata(result_dir: Path, models: tuple[str, ...] | list[str] | None = None) -> list[SampleMetadata]:
    selected_models = tuple(models) if models is not None else MODELS
    by_model = _parse_model_files(result_dir, selected_models)
    key_sets = {model: set(rows) for model, rows in by_model.items()}
    all_keys = set().union(*key_sets.values())
    common_keys = set.intersection(*key_sets.values())

    missing = {
        model: sorted(all_keys - keys)
        for model, keys in key_sets.items()
        if all_keys - keys
    }
    if missing:
        details = "; ".join(
            f"{model}: {len(keys)} missing" for model, keys in missing.items()
        )
        raise RuntimeError(f"Model folders do not contain matching logical samples: {details}")

    rows: list[SampleMetadata] = []
    for key in sorted(common_keys):
        base = by_model[selected_models[0]][key]
        file_names = {
            model: by_model[model][key].file_name
            for model in selected_models
        }
        rows.append(
            SampleMetadata(
                sample_key=key,
                representative_file_name=base.file_name,
                source_schema=base.source_schema,
                cage_number=base.cage_number,
                sex=base.sex,
                week=base.week,
                condition=base.condition,
                id_number=base.id_number,
                pnd=base.pnd,
                date=base.date,
                partner_id=base.partner_id,
                modelA_file_name=file_names.get("modelA", ""),
                modelB_file_name=file_names.get("modelB", ""),
                exact_file_name_match=len(set(file_names.values())) == 1,
            )
        )
    return rows


def _sex_from_cage(cage_number: str) -> str:
    match = re.search(r"_([mf])_\d+$", cage_number)
    if not match:
        raise ValueError(f"Cannot parse sex from cage number: {cage_number}")
    return match.group(1)


def _logical_key(row: ParsedMetadata) -> str:
    age = f"PND{row.pnd}" if row.pnd is not None else f"{row.week}w"
    partner = row.partner_id or "NA"
    return "|".join(
        [
            row.source_schema,
            row.cage_number,
            age,
            row.condition,
            row.id_number,
            partner,
        ]
    )


def _parse_model_files(
    result_dir: Path,
    models: tuple[str, ...] | list[str] = MODELS,
) -> dict[str, dict[str, ParsedMetadata]]:
    by_model: dict[str, dict[str, ParsedMetadata]] = {}
    for model in models:
        model_dir = result_dir / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")

        parsed: dict[str, ParsedMetadata] = {}
        duplicates: dict[str, list[str]] = {}
        for path in sorted(model_dir.glob("*.csv")):
            row = parse_filename(path.name)
            key = _logical_key(row)
            if key in parsed:
                duplicates.setdefault(key, [parsed[key].file_name]).append(path.name)
            parsed[key] = row

        if duplicates:
            details = "; ".join(
                f"{key}: {names}" for key, names in sorted(duplicates.items())
            )
            raise RuntimeError(f"Duplicate logical samples in {model}: {details}")

        by_model[model] = parsed
    return by_model
