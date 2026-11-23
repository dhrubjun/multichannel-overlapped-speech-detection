"""Convert validated AMI M1 log-Mel NPZ features into memory-mappable NPY files.

Run from the repository root:
    python -m scripts.convert_m1_official_to_memmap --limit 5
    python -m scripts.convert_m1_official_to_memmap

This is a non-destructive conversion. Original NPZ files remain untouched.
"""

import argparse
import csv
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs' / 'dataset_audit' / 'official_es_is'
FEATURES = ROOT / 'outputs' / 'features'
LABELS = ROOT / 'outputs' / 'labels'
OUTPUT = ROOT / 'outputs' / 'features_m1_official_npy'
SPLITS = ('train', 'development', 'test')


def read_manifest(split):
    path = MANIFEST / f'{split}_meetings.txt'
    if not path.is_file():
        raise FileNotFoundError(path)
    meetings = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not meetings:
        raise ValueError(f'Empty manifest: {path}')
    if len(set(meetings)) != len(meetings):
        raise ValueError(f'Duplicate meetings: {path}')
    return meetings


def validate_existing(path, expected_shape):
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    if data.dtype != np.float32 or data.shape != expected_shape:
        raise ValueError(f'Existing file has wrong shape/dtype: {path}')
    # Sample and check every block for NaN/inf without loading full file at once.
    for start in range(0, len(data), 8192):
        if not np.isfinite(data[start:start+8192]).all():
            raise ValueError(f'Non-finite existing data: {path}')


def convert(meeting, overwrite=False):
    source = FEATURES / f'{meeting}_logmel_mic01.npz'
    label_path = LABELS / f'{meeting}_frame_labels.npz'
    target = OUTPUT / f'{meeting}_logmel_mic01.npy'
    temporary = OUTPUT / f'{meeting}_logmel_mic01.partial.npy'
    if not source.is_file() or not label_path.is_file():
        raise FileNotFoundError(f'{meeting}: source features or labels missing')
    with np.load(label_path, allow_pickle=False) as label_data:
        label_times = label_data['frame_times']
        labels = label_data['labels']
        mask = label_data['valid_mask']
    with np.load(source, allow_pickle=False) as data:
        x = data['features']
        times = data['frame_times']
    if x.dtype != np.float32 or x.shape != (len(labels), 80):
        raise ValueError(f'{meeting}: invalid feature array {x.shape} / {x.dtype}')
    if len(mask) != len(labels) or mask.dtype != np.bool_:
        raise ValueError(f'{meeting}: invalid validity mask')
    if not np.allclose(times, label_times, atol=1e-9, rtol=0):
        raise ValueError(f'{meeting}: feature/label timestamp mismatch')
    if not np.isfinite(x).all():
        raise ValueError(f'{meeting}: non-finite features')
    if target.exists() and not overwrite:
        validate_existing(target, x.shape)
        # Check numerical fidelity even if the NPY already exists.
        existing = np.load(target, mmap_mode='r', allow_pickle=False)
        for start in range(0, len(x), 8192):
            if not np.array_equal(existing[start:start+8192], x[start:start+8192]):
                raise ValueError(f'{meeting}: existing NPY differs from NPZ')
        return 'reused', len(x), int(mask.sum())
    if temporary.exists():
        raise FileExistsError(f'Remove/inspect stale partial output first: {temporary}')
    try:
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=x.shape)
        for start in range(0, len(x), 8192):
            output[start:start+8192] = x[start:start+8192]
        output.flush()
        del output
        validate_existing(temporary, x.shape)
        check = np.load(temporary, mmap_mode='r', allow_pickle=False)
        for start in range(0, len(x), 8192):
            if not np.array_equal(check[start:start+8192], x[start:start+8192]):
                raise ValueError(f'{meeting}: NPY data differs from original')
        del check
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return 'converted', len(x), int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='Process at most N meetings for a smoke test')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    all_meetings = [(split, m) for split in SPLITS for m in read_manifest(split)]
    names = [m for _, m in all_meetings]
    if len(names) != 96 or len(set(names)) != 96:
        raise ValueError(f'Expected 96 distinct meetings, found {len(names)} entries and {len(set(names))} unique')
    selected = all_meetings[:args.limit] if args.limit else all_meetings
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (split, meeting) in enumerate(selected, start=1):
        try:
            status, frames, valid = convert(meeting)
            print(f'[{i}/{len(selected)}] {split:11s} {meeting:8s} {status:9s} frames={frames:,} valid={valid:,}', flush=True)
            rows.append({'split': split, 'meeting': meeting, 'status': status, 'frames': frames, 'valid_frames': valid, 'error': ''})
        except Exception as exc:
            print(f'FAILED {meeting}: {exc}', flush=True)
            rows.append({'split': split, 'meeting': meeting, 'status': 'failed', 'frames': '', 'valid_frames': '', 'error': str(exc)})
            break
    report = MANIFEST / ('m1_npy_conversion_smoke.csv' if args.limit else 'm1_npy_conversion_audit.csv')
    with report.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['split','meeting','status','frames','valid_frames','error'])
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row['status']=='failed' for row in rows)
    print(f'Completed {len(rows)}/{len(selected)}; failures={failures}; report={report}')
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
