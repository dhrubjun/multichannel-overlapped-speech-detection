# M2.4: Log-Mel + IPD TCN Training Results

## Objective

Train a three-class temporal convolutional network using combined log-Mel and inter-channel phase difference (IPD) features, and assess whether spatial information improves overlapping-speech detection.

## Model Configuration

- Input: 2,136 features per frame (80 log-Mel + 2,056 masked IPD).
- Parameters: 260,483.
- Five residual TCN blocks, dilations 1, 2, 4, 8, 16.
- Hidden channels: 64.
- Receptive field: 125 frames.
- Output: Three classes (no annotated speech, single speaker, overlap).
- Training meetings: ES2008a, ES2008b, ES2008c.
- Development meeting: ES2003a.
- Sequence length: 200 frames.
- Batch size: 8.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Maximum epochs: 20.
- Early-stopping patience: 5.
- Checkpoint selection: Development overlap average precision.
- Random seed: 42.

## Training Results

| Metric | M2-U | M2-W |
|---|---:|---:|
| Class weighting | None | Square-root inverse frequency |
| Epochs completed | 6 | 8 |
| Best epoch | 1 | 3 |
| Overlap AP | 0.110748 | 0.130347 |
| Overlap precision | 0.1476 | 0.1285 |
| Overlap recall | 0.1456 | 0.1817 |
| Overlap F1 | 0.1466 | 0.1506 |
| Macro-F1 | 0.6139 | 0.6133 |

## Error Analysis

M2-U correctly identified 323 of 2,218 overlap frames, whereas M2-W correctly identified 403.

In M2-W, 1,706 reference overlap frames were classified as single-speaker speech, making single-speaker versus overlap confusion the dominant error.

Class weighting improved overlap AP and recall but did not resolve this confusion.

## Comparison With M1

The previously documented M1 development overlap AP values were 0.1470 for the unweighted baseline and 0.1663 for the weighted baseline.

The corresponding M2 values were 0.1107 and 0.1303.

Under the current configuration, adding IPD features did not improve development overlap AP.

## Interpretation and Limitations

The M2 model uses substantially more input features and parameters than M1.

Possible explanations for the weaker development performance include redundant or noisy phase features, the high-dimensional input projection, limited acoustic diversity in the training data and differences between training and development meetings.

These explanations have not been independently tested.

The experiments used one random seed. Further analysis is required before drawing general conclusions about spatial features.

## Next Steps

- Re-evaluate all selected M1 and M2 checkpoints under a common development protocol.
- Compare class confusion matrices.
- Generate overlap precision-recall curves.
- Examine overlap probability distributions and decision thresholds.
- Determine whether feature reduction or separate spectral/spatial processing warrants further experiments.

The held-out test set remains unused.