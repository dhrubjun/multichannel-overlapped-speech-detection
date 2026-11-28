# M2 Compact (336 Features): Official AMI Development Results

**Project:** Deep Learning for Multichannel Overlapped Speech Detection Using Spectral and Spatial Audio Features  
**Model:** M2 Compact, 80 log-Mel + 256 inter-microphone phase-difference (IPD) features + SpeechTCN  
**Partition:** Official AMI Edinburgh/Idiap group split, 72 train / 12 development / 12 untouched test recordings  
**Experiment:** Seeds 42, 43 and 44, complete full-data CPU training  
**Status:** Three training runs complete; development-only results; held-out test not evaluated  
**Date:** 28 November 2026

## 1. Research question

Does a compact spatial representation improve 10 ms frame-wise detection of overlap compared with the single-channel log-Mel baseline, while using far fewer input features than the original 2,136-dimensional M2 model?

Speech-state classes: `0` no annotated speech, `1` exactly one annotated speaker, `2` at least two simultaneously active annotated speakers. Metrics exclude frames with `valid_mask=False`.

## 2. Data and features

| Setting | Value |
|---|---|
| Source | AMI Array 1, Edinburgh (ES) and Idiap (IS) meetings |
| Available recordings | 96; absent: IS1002a, IS1003b, IS1005d, IS1007d |
| Train/dev/test | 72 / 12 / 12, session groups kept together |
| Development meetings | ES2003a–d, ES2011a–d, IS1008a–d |
| Valid development frames | 1,920,947 |
| Log-Mel | 80 features, microphone 01 |
| IPD | 256 features, 2 microphone pairs × 64 frequency bins × cosine/sine |
| Combined input | 336 float32 features per frame |
| Feature location | `outputs/features/m2_compact_336/{meeting}_m2_336.npy` |
| Labels | `outputs/labels/{meeting}_frame_labels.npz` |
| Frame step | 10 ms |
| Normalization | Log-Mel dimensions 0:80 only, train-only statistics; IPD left unstandardized to preserve masked zero values |
| Statistics file | `outputs/dataset_audit/official_es_is/m1_logmel_train_stats.npz` |
| Normalization provenance | 12,147,428 valid training frames |

The feature-header audit passed for **96/96** compact arrays: shape `(frames, 336)`, dtype `float32`, corresponding JSON metadata. A separate structural audit passed for feature-to-label frame alignment and split membership (72/12/12; zero reported errors). These checks do **not** constitute a full numerical integrity scan of all feature values or a verification that acoustic annotations are correct.

**Reproducibility note:** The training configurations identify the compact representation as 80 log-Mel + 256 IPD but do not record the actual two microphone-pair identities, exact selected FFT bin indices, energy-mask threshold, or feature-extraction script commit. These must be copied from the extraction metadata and recorded before publication; do not assume them from the dimensionality alone.

## 3. Architecture and training

| Setting | Value |
|---|---|
| Network | `SpeechTCN`, same family as M1 |
| Parameters | 145,283 |
| Sequence length | 200 frames (2 seconds) |
| Train / dev strides | 100 / 200 frames |
| Training / dev sequences | 126,727 / 9,951 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Batch size | 8 |
| Max epochs / patience | 20 / 5 |
| Loss | Unweighted masked cross-entropy |
| Gradient clipping | Max global norm 1.0 |
| Seeds | 42 / 43 / 44 |
| Checkpoint criterion | Highest pooled development overlap AP |
| Hardware/software | CPU, PyTorch 2.14.1+cpu per training configurations |
| Test split | Not loaded for training or development checkpoint selection |

Features were read from memory-mapped NumPy arrays. The resulting classifier predicts one of the three speech states for each 10 ms frame. Overlap F1/precision/recall below use the argmax over three logits, whereas overlap AP uses scores for class 2.

## 4. Results at the selected checkpoints

These scores are from **each seed's maximum development AP checkpoint**, not necessarily the last epoch.

| Seed | Best epoch | Epochs run | Overlap AP | Precision | Recall | Overlap F1 | Macro-F1 | Dev loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 15 | 0.6532 | 0.7598 | 0.4925 | 0.5976 | 0.7656 | 0.3388 |
| 43 | 14 | 19 | 0.6758 | 0.7431 | 0.5375 | 0.6238 | 0.7822 | 0.3074 |
| 44 | 10 | 15 | 0.6693 | 0.7330 | 0.5359 | 0.6191 | 0.7794 | 0.3075 |

### 4.1 Three-seed mean ± sample standard deviation

| Metric | Mean ± sample SD |
|---|---:|
| Overlap AP | **0.6661 ± 0.0117** |
| Overlap precision | **0.7453 ± 0.0136** |
| Overlap recall | **0.5220 ± 0.0255** |
| Overlap F1 | **0.6135 ± 0.0140** |
| Macro-F1 | **0.7757 ± 0.0089** |
| Development loss | **0.3179 ± 0.0181** |
| Class 0 F1 (no speech) | **0.7940 ± 0.0083** |
| Class 1 F1 (single speaker) | **0.9196 ± 0.0045** |
| Class 2 F1 (overlap) | **0.6135 ± 0.0140** |

The sample SD describes variation among only three training seeds, not confidence bounds for generalization to unseen meetings. Each selected checkpoint was evaluated on 1,920,947 valid development frames.

### 4.2 Observations

- The best development overlap AP is **0.6758 (seed 43, epoch 14)**; seed 43 also has the best argmax overlap F1 among the three selected checkpoints (**0.6238**).
- Development AP increased from the early epochs and fluctuated later. Runs stopped after 15, 19 and 15 epochs, respectively, under patience 5.
- Mean overlap recall (**0.5220**) remains lower than mean precision (**0.7453**). Many annotated overlap frames are still predicted as another class.
- Macro-F1 is **0.7757 ± 0.0089**. Per-class F1 is calculated from each checkpoint confusion matrix before averaging across seeds.

## 5. Matched M1 baseline comparison

M1 results are already documented in `docs/m1_official_expanded_baseline_results.md` and merged with training/evaluation scripts in [PR #44](https://github.com/dhrubjun/multichannel-overlapped-speech-detection/pull/44). Both experiments use the same 72/12/12 meeting partition, train-only log-Mel normalization provenance, 200-frame sequences, optimizer family, and development checkpoint-selection metric. M1 was run on Intel XPU and compact M2 on CPU, so numerical backend differences cannot be ruled out.

| Metric | M1 log-Mel (mean ± SD) | M2 Compact (mean ± SD) | Difference of means (M2 − M1) |
|---|---:|---:|---:|
| Overlap AP | 0.5735 ± 0.0080 | 0.6661 ± 0.0117 | +0.0926 |
| Overlap precision | 0.6917 ± 0.0525 | 0.7453 ± 0.0136 | +0.0536 |
| Overlap recall | 0.4187 ± 0.0479 | 0.5220 ± 0.0255 | +0.1033 |
| Overlap F1 | 0.5187 ± 0.0243 | 0.6135 ± 0.0140 | +0.0948 |
| Macro-F1 | 0.7411 ± 0.0088 | 0.7757 ± 0.0089 | +0.0346 |

M1 uses **128,899** trainable parameters; compact M2 uses **145,283**, an increase of **16,384** (about **12.7%**). The difference in parameters means this is a matched-data comparison, **not** a parameter-controlled isolation of IPD information. Additional controls are needed before attributing gains specifically to the spatial cues.

The M1 figures above are taken from the previously merged report and are rounded to four decimals. Improvements listed are differences of reported mean scores, not uncertainty estimates or significance tests. This comparison is development-only, not held-out test performance.

## 6. Reproducibility and artifacts

Relevant source files intended for the compact experiment:

```text
scripts/smoke_m2_compact_336.py
scripts/train_m2_compact_336.py
scripts/summarize_m2_compact_336.py
src/models/tcn.py
src/training/engine.py
src/training/evaluation.py
src/training/early_stopping.py
```

Experiment outputs **not for Git** (retain locally):

```text
outputs/training/m2_compact_336/full_seed42/{config.json,history.json,summary.json,best_model.pt}
outputs/training/m2_compact_336/full_seed43/{config.json,history.json,summary.json,best_model.pt}
outputs/training/m2_compact_336/full_seed44/{config.json,history.json,summary.json,best_model.pt}
```

Small, auditable results suitable for Git:

```text
docs/m2_compact_336_official_results.md
docs/results/m2_compact_336_three_seed_summary.csv
```

`best_model.pt` contains optimizer and model tensors and should remain local or in separate managed experiment storage, not the source repository. The uploaded checkpoint bytes were not used to generate metrics; the report uses the associated histories, configurations and summaries.

**Suggested run:** `python -m scripts.train_m2_compact_336 --seed 42 --epochs 20 --batch-size 8 --patience 5 --cpu-threads 4`, then repeat with seeds 43 and 44. Do not rerun using a populated output directory because the trainer protects against overwrites.

## 7. Limitations and next steps

1. Reported scores were used for checkpoint selection on the same development partition, and must not be treated as independent test performance.
2. Averages over three seeds do not quantify meeting-to-meeting generalization uncertainty. Perform a per-meeting comparison against M1 on the 12 development meetings.
3. True spatial information has not been isolated from model-capacity effects. Recommended controls: random-IPD input of the same dimension and a parameter-matched spectral-only model.
4. The precise compact IPD bin selection and microphone pairs must be recorded from extraction metadata before claiming a specific physical interpretation.
5. Complete the original 2,136-feature M2 run on the **same** official split and seeds, then evaluate controlled 4-pair × 32-bin features.
6. Preserve the 12 test recordings for the final predeclared evaluation only.

## 8. Conclusion

Across three seeds, compact 336-feature M2 achieved pooled development overlap **AP 0.6661 ± 0.0117**, **argmax F1 0.6135 ± 0.0140**, and **macro-F1 0.7757 ± 0.0089**, exceeding the previously documented official-split M1 development means. These results establish a compact spatial-input benchmark, subject to the controls and held-out evaluation described above.
