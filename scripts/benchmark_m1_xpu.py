"""Benchmark real AMI M1 training batches on CPU or Intel XPU.

Place in scripts/ and run from repository root:
    python -m scripts.benchmark_m1_xpu --device xpu --batches 100

No checkpoints are saved; no original features or model files are modified.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.m1_official_dataset import M1OfficialDataset
from src.models.tcn import SpeechTCN
from src.training.engine import train_one_epoch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'dataset_audit' / 'official_es_is'


def sync(device):
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['xpu','cuda','cpu'], default='xpu')
    p.add_argument('--batches', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.batches, args.warmup, args.batch_size) < 1:
        p.error('batches, warmup, batch-size must be positive')
    if args.device == 'xpu' and not torch.xpu.is_available():
        p.error('Intel XPU is not available')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is not available')

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    meetings = [line.strip() for line in (AUDIT/'train_meetings.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(meetings) != 72 or len(set(meetings)) != 72:
        raise RuntimeError('Expected 72 unique training meetings')
    stats_path = AUDIT/'m1_logmel_train_stats.npz'
    with np.load(stats_path, allow_pickle=False) as data:
        if int(data['count']) != 12_147_428 or set(data['train_meetings'].tolist()) != set(meetings):
            raise RuntimeError('Normalization provenance mismatch')

    dataset = M1OfficialDataset(meetings=meetings, project_root=ROOT, sequence_length=200, stride=100, normalization_stats=stats_path)
    needed = (args.warmup + args.batches) * args.batch_size
    if needed > len(dataset):
        p.error('Requested more sequences than available')
    # Randomly sample sequences across all training meetings instead of just the first meeting.
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=needed, replace=False).tolist()
    warmup_loader = DataLoader(Subset(dataset, indices[:args.warmup * args.batch_size]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    timed_loader = DataLoader(Subset(dataset, indices[args.warmup * args.batch_size:]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SpeechTCN(input_features=80).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f'Device: {device}, torch={torch.__version__}', flush=True)
    print(f'Training windows available: {len(dataset):,}', flush=True)
    print(f'Warmup batches: {args.warmup}; timed batches: {args.batches}; batch size: {args.batch_size}', flush=True)
    print('Running warmup...', flush=True)
    train_one_epoch(model, warmup_loader, optimizer, device=device, max_grad_norm=1.0, class_weights=None)
    sync(device)

    print('Measuring real training batches...', flush=True)
    start = time.perf_counter()
    result = train_one_epoch(model, timed_loader, optimizer, device=device, max_grad_norm=1.0, class_weights=None)
    sync(device)
    elapsed = time.perf_counter() - start
    batches = result['batches']
    if batches != args.batches:
        raise RuntimeError(f'Only processed {batches}/{args.batches} batches; check masks')
    rate = batches / elapsed
    full_batches = math.ceil(len(dataset) / args.batch_size)
    print('\nM1 XPU BENCHMARK RESULT')
    print(f'Processed: {batches} batches ({batches * args.batch_size} sequences)')
    print(f'Elapsed: {elapsed:.2f} s')
    print(f'Throughput: {rate:.2f} batches/s')
    print(f'Training loss: {result["loss"]:.4f}')
    print(f'Projected training-only epoch: {full_batches / rate / 60:.1f} minutes (rough estimate)')
    print('Development evaluation, checkpoint I/O, and later-epoch variability are not included.')
    if device.type == 'xpu':
        try:
            print(f'Peak XPU allocated: {torch.xpu.max_memory_allocated()/1024**3:.3f} GiB')
        except (AttributeError, RuntimeError):
            pass
    print('BENCHMARK PASSED')


if __name__ == '__main__':
    main()
