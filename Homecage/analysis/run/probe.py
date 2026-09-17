"""Refit and verify the five held-out-cage nonlinear probe folds for FigS3B."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
from threadpoolctl import threadpool_limits
from analysis.core.archive import ROOT
from analysis.core.verification import refit_probe


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/work/probe_refit')
    out = parser.parse_args().output.resolve()
    if ROOT/'analysis/work' not in out.parents:
        raise ValueError('Probe output must be a subdirectory of analysis/work')
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    with threadpool_limits(limits=1):
        report = refit_probe(out)
    (out/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
