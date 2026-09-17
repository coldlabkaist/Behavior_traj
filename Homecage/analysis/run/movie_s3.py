"""Render the retained ten-set Movie S3 compilation from source AVI recordings."""
from argparse import ArgumentParser
from pathlib import Path
import shutil
from analysis.core.archive import ROOT
from analysis.plots.movie_s3 import load_selection, render_movie


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, action='append', required=True,
                        help='Original AVI folder; repeat in search-priority order')
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/work/movie_s3/Movie_S3.mp4')
    parser.add_argument('--check-only', action='store_true', help='Verify scene choices and input paths')
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT/'analysis/work').resolve() not in output.parents or output.suffix.lower() != '.mp4':
        raise ValueError('Movie output must be an MP4 within analysis/work')
    records = load_selection(args.video_root)
    for command in ['ffmpeg', 'ffprobe']:
        if shutil.which(command) is None:
            raise RuntimeError(f'{command} is required on PATH')
    if args.check_only:
        print('40 selected windows match Final; all source recordings and pose inputs exist.')
        return
    print(render_movie(records, output))


if __name__ == '__main__':
    main()
