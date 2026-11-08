# M2-D1: Trained Model Reliance on IPD Features

## Objective

Determine whether the trained M2 weighted TCN relies on IPD features when predicting overlapped speech.

## Experimental Setup

- Model: M2-W, 2,136-dimensional log-Mel + IPD TCN.
- Checkpoint: `outputs/training/m2_weighted_v1/best_model.pt`
- Development meeting: ES2003a.
- Valid evaluation frames: 111,472.
- Training: None. The same saved checkpoint was used throughout.

Three conditions were evaluated:

1. Original log-Mel and IPD features.
2. Original log-Mel with all IPD features set to zero.
3. Original log-Mel with IPD frames temporally shuffled within each sequence.

## Results

| Metric | Original | Zero IPD | Shuffled IPD |
|---|---:|---:|---:|
| Overlap AP | 0.130347 | 0.027238 | 0.068435 |
| Argmax overlap F1 | 0.150570 | 0.049349 | 0.099026 |
| Mean overlap score, true overlap | 0.224230 | 0.322937 | 0.238268 |
| Mean overlap score, non-overlap | 0.056838 | 0.231862 | 0.087985 |

Relative to the original condition, zeroing IPD reduced AP by 79.10% and changed 38.49% of argmax predictions.

Shuffling IPD reduced AP by 47.50% and changed 13.24% of argmax predictions.

## Interpretation

The trained M2-W model is strongly sensitive to the IPD feature group.

Removing or temporally disrupting IPD substantially reduces development overlap detection performance.

The model's learned predictions depend on the spatial input, but this does not demonstrate a benefit over an independently trained spectral-only model.

The previously trained M1-W model remains stronger on the development set.

## Limitations

- Zeroing or shuffling IPD produces inputs outside the original training distribution.
- The experiment measures model sensitivity rather than the causal benefit of adding IPD during training.
- The temporal shuffle may disrupt multiple acoustic properties simultaneously.
- Results were obtained from one trained checkpoint and one development meeting.

## Conclusion

The initial M2 model does not ignore IPD. Subsequent work should investigate microphone geometry, phase consistency, frequency selection and spatial fusion before concluding that the spatial features themselves are ineffective.

## Next Experiment

M2-D2: Validate microphone geometry and frequency-dependent phase behaviour.