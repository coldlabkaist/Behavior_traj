"""Extract and verify the frozen DAE latent with the restored loader settings."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import torch
from analysis.common import choose_device
from analysis.core.archive import ROOT
from analysis.core.verification import extract_and_verify_latent


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/work/latent_extraction')
    parser.add_argument('--device', default='')
    args = parser.parse_args()
    out = args.output.resolve()
    if ROOT/'analysis/work' not in out.parents:
        raise ValueError('Extraction output must be a subdirectory of analysis/work')
    out.mkdir(parents=True, exist_ok=True)
    destination = out/'latent_windows.pt'
    if destination.exists():
        raise FileExistsError('Use a new output directory to preserve the previous extraction')
    os.chdir(ROOT)
    torch.set_num_threads(4)
    report = extract_and_verify_latent(destination, choose_device(args.device))
    (out/'extraction_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not report['tensor_values_identical']:
        raise RuntimeError('Candidate latent differs from Final; inspect report before downstream use')


if __name__ == '__main__':
    main()
