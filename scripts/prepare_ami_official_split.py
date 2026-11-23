"""Audit AMI Array1 audio and export the official ES/IS scenario-only split.
Run from the repository root: python prepare_ami_official_split.py
No audio is modified; no model is trained.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import wave
from collections import Counter
from pathlib import Path

TRAIN_GROUPS = [
    'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010',
    'ES2012','ES2013','ES2015','ES2016',
    'IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006','IS1007',
]
DEV_GROUPS = ['ES2003', 'ES2011', 'IS1008']
TEST_GROUPS = ['ES2004', 'ES2014', 'IS1009']
# Scenario meetings not in the published AMI meeting set / with no available Array1 audio.
KNOWN_UNAVAILABLE = {'IS1002a', 'IS1003b', 'IS1005d', 'IS1007d'}
DEFAULT_ROOT = Path(r'C:\Master 1st Sem\Deep Learning\Project\amicorpus')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='AMI corpus directory')
    p.add_argument('--out', type=Path, default=Path('outputs/dataset_audit/official_es_is'))
    args = p.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    if not root.is_dir():
        print(f'ERROR: dataset directory not found: {root}', file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    group_map = {'train': TRAIN_GROUPS, 'development': DEV_GROUPS, 'test': TEST_GROUPS}
    group_owner = {grp: split for split, gs in group_map.items() for grp in gs}
    if len(group_owner) != sum(map(len, group_map.values())):
        raise RuntimeError('A recording-session group appears in more than one split')

    all_expected = {g + suffix for g in group_owner for suffix in 'abcd'} - KNOWN_UNAVAILABLE
    found = {d.name for d in root.iterdir() if d.is_dir() and re.fullmatch(r'(ES|IS)\d{4}[a-d]', d.name)}
    missing = sorted(all_expected - found)
    unexpected = sorted(found - all_expected)
    problems = []
    rows = []
    for split, groups in group_map.items():
        for group in groups:
            for suffix in 'abcd':
                meeting = group + suffix
                if meeting in KNOWN_UNAVAILABLE:
                    continue
                audio_dir = root / meeting / 'audio'
                wav_paths = [audio_dir / f'{meeting}.Array1-{i:02d}.wav' for i in range(1, 9)]
                sample_rates, sampwidths, channels, frame_counts = set(), set(), set(), set()
                errors = []
                sizes = []
                for wav_path in wav_paths:
                    if not wav_path.is_file():
                        errors.append(f'Missing {wav_path.name}')
                        continue
                    try:
                        file_size = wav_path.stat().st_size
                        sizes.append(file_size)
                        with wave.open(str(wav_path), 'rb') as wav:
                            sample_rates.add(wav.getframerate())
                            sampwidths.add(wav.getsampwidth())
                            channels.add(wav.getnchannels())
                            frame_counts.add(wav.getnframes())
                            declared_payload = wav.getnframes() * wav.getsampwidth() * wav.getnchannels()
                            # Allows WAV headers with extra chunks by checking only an obvious truncation.
                            if file_size < declared_payload:
                                errors.append(f'{wav_path.name}: smaller than header-declared sample payload')
                    except (OSError, EOFError, wave.Error) as ex:
                        errors.append(f'{wav_path.name}: WAV header read failed: {ex}')
                if len(sample_rates) != 1 or (sample_rates and next(iter(sample_rates)) != 16000):
                    errors.append(f'Unexpected/inconsistent sample rate: {sorted(sample_rates)}')
                if sampwidths != {2}:
                    errors.append(f'Unexpected/inconsistent sample width (bytes): {sorted(sampwidths)}')
                if channels != {1}:
                    errors.append(f'Unexpected/inconsistent channels per WAV: {sorted(channels)}')
                if len(frame_counts) != 1:
                    errors.append(f'Inconsistent sample counts: {sorted(frame_counts)}')
                good = not errors
                if not good:
                    problems.append((meeting, errors))
                rows.append({
                    'split':split, 'meeting':meeting, 'group':group,
                    'files_found':sum(p.is_file() for p in wav_paths),
                    'sample_rate_hz':next(iter(sample_rates)) if len(sample_rates) == 1 else '',
                    'duration_seconds':round(next(iter(frame_counts))/16000, 3) if len(frame_counts)==1 else '',
                    'audio_ok':good,
                    'problems':' | '.join(errors),
                })

    # Export deterministic splits regardless of the validation results, with validation status recorded.
    for split in group_map:
        ids = [r['meeting'] for r in rows if r['split'] == split]
        (out / f'{split}_meetings.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
    (out / 'split.json').write_text(json.dumps({
        'name':'AMI official scenario-only partition restricted to Edinburgh and Idiap Array 1',
        'source':'https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml',
        'root':str(root),
        'known_unavailable':sorted(KNOWN_UNAVAILABLE),
        'groups':group_map,
        'meetings':{split:[r['meeting'] for r in rows if r['split']==split] for split in group_map},
    },indent=2)+'\n',encoding='utf-8')
    with (out / 'audio_audit.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count = Counter(r['split'] for r in rows)
    print('AMI OFFICIAL ES/IS ARRAY 1 DATASET AUDIT')
    print('Root:',root)
    print('Output:',out)
    print(f"Expected meetings: {len(all_expected)} (train={count['train']}, dev={count['development']}, test={count['test']})")
    print('Known unavailable:', ', '.join(sorted(KNOWN_UNAVAILABLE)))
    print('Actual matching meeting directories:',len(found))
    print('Missing expected directories:',', '.join(missing) if missing else 'None')
    print('Unexpected ES/IS directories:',', '.join(unexpected) if unexpected else 'None')
    print('Audio passing header and consistency checks:',sum(r['audio_ok'] for r in rows),'/',len(rows))
    for meeting, issues in problems[:12]:
        print('FAIL',meeting,':','; '.join(issues))
    if len(problems)>12:
        print('Other failures:',len(problems)-12,'(see CSV)')
    if missing or unexpected or problems:
        print('AUDIT FAILED: Inspect audio_audit.csv; do not start model training yet.')
        return 1
    print('AUDIT PASSED: manifests generated. Next: validate annotations and generate frame labels.')
    print('Note: this header audit does not decode the full audio or verify actual transcription quality.')
    return 0


if __name__=='__main__':
    sys.exit(main())
