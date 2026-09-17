"""Project-relative locations used by the paper execution commands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'output' / 'Final'
REPRODUCED = ROOT / 'output' / 'reproduced'
DATA_ROOT = ROOT.parent / 'data'
RAW_DATA = {
    '3chamber': DATA_ROOT / 'csv/Fig5BC_FigS8ABCD',
    'mom_pup': DATA_ROOT / 'csv/Fig5EFH',
    'oft': DATA_ROOT / 'csv/FigS10AB',
}


def resolve_input_path(value) -> Path:
    """Read relocated raw files while preserving paths in frozen manifests."""
    path = Path(str(value))
    relative = path
    if path.is_absolute():
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return path
    if len(relative.parts) >= 2 and relative.parts[0] == 'data':
        key = relative.parts[1].lower()
        if key in RAW_DATA:
            return RAW_DATA[key].joinpath(*relative.parts[2:])
    return path if path.is_absolute() else ROOT / path


def relative_input_path(value) -> str:
    path = Path(value).resolve()
    for key, base in RAW_DATA.items():
        try:
            tail = path.relative_to(base)
        except ValueError:
            continue
        label = 'OFT' if key == 'oft' else key
        return (Path('data') / label / tail).as_posix()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def panel_input(panel: str, category: str, name: str) -> Path:
    return FINAL / panel / category / name
