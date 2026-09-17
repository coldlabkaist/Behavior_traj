"""Resolve frozen dataset identifiers to the separately distributed raw data."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT.parent / 'data/csv/Fig4B-G_FigS3-S7'


def resolve_data_path(value) -> Path:
    path = Path(str(value))
    relative = path
    if path.is_absolute():
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            return path
    if len(relative.parts) >= 2 and relative.parts[0] == 'data':
        if relative.parts[1] in {'cont', 'experiments', 'reference', 'roi'}:
            return RAW_ROOT.joinpath(*relative.parts[1:])
    return path if path.is_absolute() else PROJECT_ROOT / path
