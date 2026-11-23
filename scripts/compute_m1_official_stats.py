"""Compute 80-band log-Mel normalization statistics from official AMI train split.

Run from the repository root:
    python -m scripts.compute_m1_official_stats

Only valid training frames contribute. The original baseline stats are untouched.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / 'outputs' / 'dataset_audit' / 'official_es_is'
FEATURE_DIR = ROOT / 'outputs' / 'features'
LABEL_DIR = ROOT / 'outputs' / 'labels'
MANIFEST = AUDIT_DIR / 'train_meetings.txt'
OUTPUT = AUDIT_DIR / 'm1_logmel_train_stats.npz'
EXPECTED_MEETINGS = 72
EXPECTED_VALID = 12_147_428
N_MELS = 80


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    meetings = [line.strip() for line in MANIFEST.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(meetings) != EXPECTED_MEETINGS or len(set(meetings)) != len(meetings):
        raise ValueError(f'Expected {EXPECTED_MEETINGS} unique training meetings, got {len(meetings)}')
    for split in ('development', 'test'):
        path = AUDIT_DIR / f'{split}_meetings.txt'
        if not path.is_file():
            raise FileNotFoundError(path)
        other = {line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
        overlap = set(meetings) & other
        if overlap:
            raise ValueError(f'Train/{split} overlap: {sorted(overlap)}')

    if OUTPUT.exists():
        raise FileExistsError(f'Will not overwrite existing normalization: {OUTPUT}')

    # Parallel Welford merge of per-meeting moments, in float64.
    count = 0
    mean = np.zeros(N_MELS, dtype=np.float64)
    m2 = np.zeros(N_MELS, dtype=np.float64)
    print('OFFICIAL AMI M1 TRAIN-ONLY NORMALIZATION', flush=True)
    for index, meeting in enumerate(meetings, 1):
        feature_path = FEATURE_DIR / f'{meeting}_logmel_mic01.npz'
        label_path = LABEL_DIR / f'{meeting}_frame_labels.npz'
        with np.load(feature_path, allow_pickle=False) as f:
            x = f['features']
            feature_times = f['frame_times']
            if 'n_mels' in f and int(f['n_mels']) != N_MELS:
                raise ValueError(f'{meeting}: wrong number of mel bands in metadata')
        with np.load(label_path, allow_pickle=False) as f:
            mask = f['valid_mask']
            labels = f['labels']
            label_times = f['frame_times']
        if x.ndim != 2 or x.shape[1] != N_MELS:
            raise ValueError(f'{meeting}: wrong feature shape {x.shape}')
        if mask.dtype != np.bool_ or len(mask) != len(labels) or len(mask) != len(x):
            raise ValueError(f'{meeting}: labels/mask/features mismatch')
        if not np.allclose(feature_times, label_times, rtol=0, atol=1e-9):
            raise ValueError(f'{meeting}: timestamp mismatch')
        if not np.isfinite(x).all():
            raise ValueError(f'{meeting}: nonfinite feature values')
        if not np.isin(labels[mask], [0, 1, 2]).all():
            raise ValueError(f'{meeting}: invalid valid-frame labels')
        valid = x[mask].astype(np.float64, copy=False)
        n = len(valid)
        if n == 0:
            raise ValueError(f'{meeting}: no valid frames')
        local_mean = valid.mean(axis=0)
        local_m2 = np.sum((valid - local_mean) ** 2, axis=0, dtype=np.float64)
        if count == 0:
            mean, m2, count = local_mean, local_m2, n
        else:
            new_count = count + n
            delta = local_mean - mean
            mean += delta * (n / new_count)
            m2 += local_m2 + delta * delta * (count * n / new_count)
            count = new_count
        print(f'[{index:02d}/{len(meetings)}] {meeting}: {n:,} valid frames', flush=True)

    if count != EXPECTED_VALID:
        raise ValueError(f'Expected {EXPECTED_VALID:,} valid train frames; got {count:,}. Nothing saved.')
    std = np.sqrt(m2 / count)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 1e-8).any():
        raise ValueError('Nonfinite or near-zero normalization parameters')
    # Use same field names as the existing SpeechSequenceDataset loader.
    np.savez_compressed(OUTPUT, mean=mean.astype(np.float32), std=std.astype(np.float32),
                        count=np.int64(count), train_meetings=np.array(meetings),
                        n_mels=np.int64(N_MELS), split_name=np.array('official_es_is_72_12_12'))
    print('\nNORMALIZATION PASSED')
    print(f'Training meetings: {len(meetings)}')
    print(f'Valid training frames: {count:,}')
    print(f'Mean range: [{mean.min():.6f}, {mean.max():.6f}]')
    print(f'Std range: [{std.min():.6f}, {std.max():.6f}]')
    print(f'Saved: {OUTPUT}')


if __name__ == '__main__':
    main()
