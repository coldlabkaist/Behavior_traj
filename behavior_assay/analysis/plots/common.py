from pathlib import Path

def rename_bundle(directory: Path, old: str, new: str) -> None:
    for ext in ['png', 'svg']:
        source = directory / f'{old}.{ext}'
        if source.exists() and old != new:
            source.replace(directory / f'{new}.{ext}')
