"""Batch M1 extraction for official AMI ES/IS manifests.

Place in scripts/ and run: python -m scripts.batch_extract_m1_official
Uses the existing scripts.extract_logmel_features module unchanged.
Never overwrites existing feature archives.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / 'outputs' / 'dataset_audit' / 'official_es_is'
FEATURE_DIR = ROOT / 'outputs' / 'features'
LABEL_DIR = ROOT / 'outputs' / 'labels'
REPORT = MANIFEST_DIR / 'm1_feature_extraction_audit.csv'
SPLITS = ('train', 'development', 'test')
EXPECTED = {'train': 72, 'development': 12, 'test': 12}


def meeting_list(split):
    path = MANIFEST_DIR / f'{split}_meetings.txt'
    if not path.exists():
        raise FileNotFoundError(f'Missing manifest: {path}')
    meetings = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(meetings) != EXPECTED[split] or len(meetings) != len(set(meetings)):
        raise ValueError(f'{split}: expected {EXPECTED[split]} unique meetings, got {len(meetings)}')
    return meetings


def validate(meeting):
    feature_path = FEATURE_DIR / f'{meeting}_logmel_mic01.npz'
    label_path = LABEL_DIR / f'{meeting}_frame_labels.npz'
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    with np.load(label_path, allow_pickle=False) as d:
        label_times = d['frame_times']
        labels = d['labels']
        mask = d['valid_mask']
        n = len(labels)
        valid = int(np.count_nonzero(mask))
        if mask.shape != (n,) or label_times.shape != (n,):
            raise ValueError('Inconsistent label and mask shapes')
    with np.load(feature_path, allow_pickle=False) as d:
        x = d['features']
        times = d['frame_times']
        if x.shape != (n, 80) or times.shape != (n,):
            raise ValueError(f'Unexpected feature/timestamp shapes: {x.shape}, {times.shape}, expected ({n}, 80)')
        if not np.isfinite(x).all():
            raise ValueError('Non-finite feature values')
        if not np.allclose(times, label_times, rtol=0, atol=1e-9):
            raise ValueError('Feature/label timestamps differ')
    return n, valid


def write_report(rows):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=('split', 'meeting', 'status', 'frames', 'valid_frames', 'error'))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Extract M1 features using official AMI manifests')
    ap.add_argument('--split', choices=(*SPLITS, 'all'), default='all')
    ap.add_argument('--limit', type=int, help='Process at most this many meetings (for smoke tests)')
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive')

    # Preflight all manifests to prevent split overlap.
    manifests = {name: meeting_list(name) for name in SPLITS}
    all_ids = [m for ids in manifests.values() for m in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Meetings overlap between split manifests')

    selected = [(s, m) for s in SPLITS if args.split in ('all', s) for m in manifests[s]]
    if args.limit:
        selected = selected[:args.limit]

    rows = []
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for idx, (split, meeting) in enumerate(selected, 1):
            path = FEATURE_DIR / f'{meeting}_logmel_mic01.npz'
            print(f'[{idx}/{len(selected)}] {split} {meeting}', flush=True)
            status = 'validated_existing'
            try:
                if not path.exists():
                    status = 'extracted'
                    subprocess.run(
                        [sys.executable, '-m', 'scripts.extract_logmel_features', '--meeting', meeting],
                        cwd=ROOT, check=True,
                    )
                n, valid = validate(meeting)
                rows.append(dict(split=split, meeting=meeting, status=status, frames=n, valid_frames=valid, error=''))
                print(f'PASS {meeting}: {n:,} frames, {valid:,} valid ({status})', flush=True)
            except Exception as ex:
                rows.append(dict(split=split, meeting=meeting, status='failed', frames='', valid_frames='', error=str(ex)))
                print(f'FAIL {meeting}: {ex}', file=sys.stderr, flush=True)
                if status == 'extracted' and path.exists():
                    print(f'WARNING: newly created output needs inspection: {path}', file=sys.stderr)
                # Stop rather than silently training with incomplete data.
                break
    finally:
        write_report(rows)

    failures = [r for r in rows if r['status'] == 'failed']
    print(f'Completed {len(rows)}/{len(selected)} meetings; failures={len(failures)}. Report: {REPORT}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
