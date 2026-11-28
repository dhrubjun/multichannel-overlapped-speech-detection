"""Summarize best-AP checkpoints for compact M2, using saved JSON artifacts.

Run from repository root:
    python -m scripts.summarize_m2_compact_336

Does not load .pt checkpoints or touch held-out test recordings.
"""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'outputs' / 'training' / 'm2_compact_336'
OUTPUT = ROOT / 'docs' / 'results' / 'm2_compact_336_three_seed_summary.csv'
METRICS = ('overlap_ap', 'overlap_precision', 'overlap_recall', 'overlap_f1', 'macro_f1', 'dev_loss')


def classwise_f1(cm):
    scores = []
    for k in range(3):
        tp = int(cm[k][k])
        support = sum(int(x) for x in cm[k])
        predicted = sum(int(cm[i][k]) for i in range(3))
        d = support + predicted
        scores.append(2 * tp / d if d else 0.0)
    return scores


def main():
    results = []
    baseline = None
    for seed in (42, 43, 44):
        folder = RUNS / f'full_seed{seed}'
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        if config['seed'] != seed or config['smoke_test'] or summary['smoke_test']:
            raise ValueError(f'{seed}: wrong experiment configuration')
        if config['input_features'] != 336 or config['parameters'] != 145283:
            raise ValueError(f'{seed}: unexpected model dimensions')
        if len(config['train_meetings']) != 72 or len(config['dev_meetings']) != 12:
            raise ValueError(f'{seed}: wrong split sizes')
        if set(config['train_meetings']) & set(config['dev_meetings']):
            raise ValueError(f'{seed}: training/development data leakage')
        comparable = tuple(json.dumps(config[k], sort_keys=True) for k in (
            'train_meetings', 'dev_meetings', 'train_windows', 'dev_windows',
            'batch_size', 'learning_rate', 'weight_decay', 'class_weighting',
            'normalization_train_frames', 'input_features', 'parameters',
            'sequence_length', 'train_stride', 'dev_stride'))
        if baseline is None:
            baseline = comparable
        elif baseline != comparable:
            raise ValueError(f'{seed}: training configurations differ')
        if len(history) != summary['epochs_completed']:
            raise ValueError(f'{seed}: history length mismatch')
        best = next((x for x in history if x['epoch'] == summary['best_epoch']), None)
        if best is None or abs(best['overlap_ap'] - summary['best_overlap_ap']) > 1e-9:
            raise ValueError(f'{seed}: checkpoint selection mismatch')
        f1 = classwise_f1(best['confusion_matrix'])
        row = dict(seed=seed, best_epoch=summary['best_epoch'],
                   epochs_completed=summary['epochs_completed'])
        row.update({k: float(best[k]) for k in METRICS})
        row['dev_valid_frames'] = int(best['dev_valid_frames'])
        row.update({f'class{i}_f1': f1[i] for i in range(3)})
        if row['dev_valid_frames'] != 1_920_947:
            raise ValueError(f'{seed}: unexpected development frame count')
        results.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('M2 COMPACT 336, OFFICIAL AMI DEVELOPMENT')
    for row in results:
        print(f"Seed {row['seed']}: epoch={row['best_epoch']}, "
              f"AP={row['overlap_ap']:.4f}, overlap F1={row['overlap_f1']:.4f}")
    print('\nMean ± sample SD')
    for key in METRICS + ('class0_f1', 'class1_f1', 'class2_f1'):
        values = [r[key] for r in results]
        print(f'{key:22s} {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}')
    print('Saved:', OUTPUT)


if __name__ == '__main__':
    main()
