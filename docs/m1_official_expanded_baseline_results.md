# M1: Expanded AMI Official-Group Baseline Results

**Project:** Deep Learning for Multichannel Overlapped Speech Detection Using Spectral and Spatial Audio Features  
**Model:** M1, single-channel log-Mel + temporal convolutional network (TCN)  
**Experiment:** Expanded AMI Edinburgh/Idiap official-group split  
**Status:** Three training seeds completed; detailed development evaluation reproduced; held-out test not evaluated  
**Prepared:** 23 November 2026  
**Repository:** https://github.com/dhrubjun/multichannel-overlapped-speech-detection

## 1. Objective and scope

M1 is a single-channel spectral baseline for **frame-wise speech-state classification**. At a 10 ms hop, the model predicts three classes: **0** (no annotated speech), **1** (exactly one annotated active speaker), and **2** (two or more annotated active speakers). Ground-truth labels come from AMI manual word annotations, not from a pretrained speech detector. The objective is to establish performance before introducing spatial inter-microphone phase difference (IPD) features in M2.

This report describes the expanded-dataset experiment. The earlier small-data baseline is preserved separately in `docs/m1_baseline_results.md`. Results here are development-set model-selection results, **not held-out test performance**.

## 2. Dataset and split

The AMI Edinburgh (`ES`) and Idiap (`IS`) meetings were organized by complete recording-session groups to avoid splitting suffixes `a`, `b`, `c`, and `d` from one session group across partitions. Four recordings were unavailable: `IS1002a`, `IS1003b`, `IS1005d`, and `IS1007d`. The resulting usable collection contains **96 meetings**.

| Split | Meetings | Total feature frames | Valid labelled frames | Excluded frames |
|---|---:|---:|---:|---:|
| Training | 72 | 12,669,266 | 12,147,428 | 521,838 |
| Development | 12 | 1,989,082 | 1,920,947 | 68,135 |
| Test | 12 | 2,326,223 | 2,270,411 | 55,812 |

Development meetings: `ES2003a-d`, `ES2011a-d`, and `IS1008a-d` (12 meetings). Test meetings were reserved for independent final evaluation and were not used for checkpoint selection or threshold tuning.

### 2.1 Ground-truth labels and class imbalance

Word intervals from each speaker were aligned to a 10 ms grid. The number of simultaneously active annotated speakers was capped at two: `label = min(active_speaker_count, 2)`. The pipeline produced `labels`, `valid_mask`, `speaker_activity`, `frame_times`, and `hop_seconds` for each meeting. Frames marked invalid by the annotation uncertainty mask were excluded from training loss and evaluation metrics.

| Class | Train | Development | Test |
|---|---:|---:|---:|
| 0: No annotated speech | 18.8935% | 19.6800% | 18.9818% |
| 1: Single-speaker speech | 72.1197% | 74.0846% | 72.8340% |
| 2: Overlapping speech | 8.9868% | 6.2354% | 8.1842% |

The overlap class is a minority class in every split. This motivates reporting overlap precision, recall, F1, and average precision (AP) in addition to macro-F1. The test prevalence is shown for dataset auditing only; it did not guide model development.

### 2.2 Annotation audit and recording caveats

All **96/96** label files passed structural validation. The annotation audit reported **8,472** parser issues and **6,441** entries classified as review-priority issues by the audit script; these are not equivalent to 6,441 proven incorrect acoustic labels. The issue categories are:

| Parser issue category | Count | Current handling |
|---|---:|---|
| zero-duration interval | 6,400 | Skipped; no positive time span |
| zero-duration gap | 1,490 | Skipped; no positive time span |
| positive-duration gap | 541 | Uncertainty interval; frames masked |
| end before start | 39 | Malformed interval skipped; review required |
| missing timestamp | 2 | Untimed entry skipped; cannot locate reliably |

The 39 reversed intervals comprise 37 `vocalsound` and two `gap` entries. The two untimed word entries are `Mm` and `'kay` for speaker A in `ES2008a`. We did not fabricate timestamps or reverse intervals automatically. Uncertainty-masked frames across all splits total **645,785**. The current structural checks do not establish that every manual annotation matches the audible speech.

For `ES2010d`, eight originally stereo files were verified to have identical left/right sample streams and converted to mono by retaining one channel; the originals were preserved separately. The header audit passed afterward, but known recording-quality/dropout concerns still warrant a separate acoustic audit. No meeting was removed solely because of annotation-issue counts.

## 3. Feature extraction and preprocessing

M1 uses **Array 1, microphone 01** of each meeting. The same `src/features/logmel.py` implementation was used as in the prior baseline:

| Parameter | Setting |
|---|---|
| Audio sampling rate | 16,000 Hz |
| Frame hop | 160 samples (10 ms) |
| Analysis window | 400 samples (25 ms), periodic Hann |
| FFT size | 512 |
| Filterbank | 80 HTK-scale triangular Mel filters |
| Spectral representation | Natural log of Mel-filtered power, floor `1e-10` |
| Frame alignment | Center at `(frame_index + 0.5) × 10 ms`, zero-padded at boundaries |
| Feature shape per meeting | `(number_of_label_frames, 80)` |

The original extraction completed for **96/96** meetings: **83** new extractions and **13** existing files validated and reused. Per-meeting feature shape, finite values, frame counts, and timestamp alignment were checked. Compressed `.npz` feature matrices were also converted to numerical-equivalent `.npy` arrays for true memory-mapped loading; the originals were retained.

### 3.1 Train-only normalization

For each Mel band `j`, the normalization statistics were computed from **only valid frames in the 72 training meetings**, with no development or test observations:

`z[t,j] = (x[t,j] - mean_train[j]) / std_train[j]`

The statistics used **12,147,428** valid training frames. The resulting band-mean range was **[-11.556297, -3.908592]**, and band-standard-deviation range was **[2.098210, 3.729875]**. The same 80 training-only means and standard deviations were applied to all three splits. The normalization file is `outputs/dataset_audit/official_es_is/m1_logmel_train_stats.npz`.

## 4. Network and sequence construction

M1 uses the existing noncausal residual `SpeechTCN` (`src/models/tcn.py`) with **128,899 trainable parameters**: a `1 × 1` input projection from 80 to 64 channels; five residual TCN blocks with two kernel-size-3 convolutions per block, dilations `(1, 2, 4, 8, 16)`, ReLU and dropout `0.1`; and a `1 × 1` classifier mapping 64 channels to three logits. Symmetric padding preserves temporal length. Its receptive field is **125 frames (1.25 s at 10 ms per frame)**.

The memory-mapped `M1OfficialDataset` constructs 200-frame (2 s) sequences without crossing meeting boundaries. Training uses a **100-frame stride** (50% overlap); development evaluation uses a **200-frame stride** (nonoverlapping sequences). Final short sequences are zero-padded and ignored using `valid_mask` and label `-100` for padding. Dataset validation passed with **126,727 training sequences**, **9,951 development sequences**, and **11,637 test sequences**. All three DataLoaders produced batches of shape `(8, 200, 80)`. The test loader was validated but was not used for model evaluation.

## 5. Training protocol

| Configuration | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 8 |
| Maximum epochs | 20 |
| Random seeds | 42, 43, 44 |
| Objective | Unweighted masked cross-entropy over valid frame labels |
| Gradient clipping | Max global norm = `1.0` |
| Early stopping | Patience 5 epochs, `min_delta=0` |
| Checkpoint selection | Highest **pooled development overlap AP** |
| Evaluation rule | `argmax` over three softmax classes; overlap score = softmax probability of class 2 |
| Hardware/software | Intel Arc 140T GPU; PyTorch `2.15.0+xpu` (user-reported) |

The Intel XPU backend initially failed on GPU Boolean indexing within the masked loss. The XPU path was changed to use `ignore_index=-100` rather than Boolean-selection indexing, preserving the unweighted loss definition. During development evaluation, logits were transferred to CPU before Boolean-indexed metric calculations. These backend workarounds should be committed alongside the experiment code.

The number of **training frame occurrences per epoch** exceeds the number of unique training frames because training windows overlap. The 12 development meetings are each evaluated once per nonoverlapping sequence. Training-time and detailed evaluation both use the same pooled-frame scoring convention.

## 6. Three-seed development results

Table values are from each seed’s **best-AP checkpoint**, not necessarily the final training epoch. AP is average precision for class 2 against all other classes, pooled over all valid development frames.

| Seed | Best epoch | Epochs run | Overlap AP | Overlap precision | Overlap recall | Overlap F1 | Macro-F1 | Dev loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 14 | 19 | 0.5777 | 0.7519 | 0.3683 | 0.4944 | 0.7353 | 0.3286 |
| 43 | 9 | 14 | 0.5786 | 0.6555 | 0.4636 | 0.5431 | 0.7512 | 0.3425 |
| 44 | 9 | 14 | 0.5643 | 0.6677 | 0.4241 | 0.5188 | 0.7368 | 0.3428 |

| Metric | Mean ± sample standard deviation (3 seeds) |
|---|---:|
| Overlap AP | **0.5735 ± 0.0080** |
| Overlap precision | **0.6917 ± 0.0525** |
| Overlap recall | **0.4187 ± 0.0479** |
| Overlap F1 | **0.5187 ± 0.0243** |
| Macro-F1 | **0.7411 ± 0.0088** |

The mean development overlap AP is **0.5735 ± 0.0080**. Seed 43 obtained the highest AP (**0.5786**) and the highest overlap F1 (**0.5431**) at its selected checkpoint. AP varies little across seeds; precision and recall vary more, reflecting different operating points under the fixed argmax decision rule.

### 6.1 Reproduction check

The standalone detailed evaluator loaded all three saved checkpoints, evaluated the **1,920,947 valid development frames**, and reproduced the training-time AP for each seed to within approximately `4.1e-11`, far below the configured `0.0005` tolerance. Thus the detailed reporting and training-time checkpoint metrics agree numerically. The held-out test was not evaluated.

## 7. Class-wise analysis

The following precision, recall, and F1 values are the **arithmetic means of the three seed-specific pooled development metrics**, not metrics computed from a concatenation of predictions across seeds:

| Class | Precision | Recall | F1 | Reference support (frames) |
|---|---:|---:|---:|---:|
| 0: No annotated speech | 0.8137 | 0.7690 | 0.7886 | 378,042 |
| 1: Single speaker | 0.8960 | 0.9372 | 0.9159 | 1,423,126 |
| 2: Overlap | 0.6917 | 0.4187 | 0.5187 | 119,779 |

The model performs best on single-speaker speech. For overlapping speech, **mean precision = 0.6917** and **mean recall = 0.4187**. At the argmax operating point, many actual overlap frames are predicted as single speaker, even when the model’s class-2 scores provide useful ranking information (as indicated by AP). This motivates a controlled spatial-feature comparison in M2.

### 7.1 Confusion matrix, seed 43 (best development AP)

Rows are reference labels and columns are argmax predictions; counts cover valid development frames only.

| Actual / predicted | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| Class 0: No speech | 313,015 | 64,340 | 687 |
| Class 1: Single | 92,393 | 1,302,239 | 28,494 |
| Class 2: Overlap | 1,538 | 62,714 | 55,527 |

For seed 43, **62,714 of 119,779 actual overlap frames** were classified as single-speaker speech, while **55,527** were correctly classified as overlap. This is the main observed error mode. Confusion matrices for seeds 42 and 44 are preserved in the exported `confusion_matrices.csv`.

## 8. Per-meeting development analysis

The table below reports the arithmetic mean of each metric across **three seeds**, computed separately for each of the 12 development meetings. These meeting-level APs must **not** be averaged and presented as the pooled development AP; pooled AP is obtained by globally ranking all development frames together.

| Meeting | Valid frames | Overlap prevalence | Mean overlap AP | Mean overlap F1 |
|---|---:|---:|---:|---:|
| ES2003a | 111,472 | 1.99% | 0.2525 | 0.1852 |
| ES2003b | 209,807 | 4.18% | 0.1787 | 0.1839 |
| ES2003c | 222,378 | 3.83% | 0.2767 | 0.2642 |
| ES2003d | 228,586 | 5.15% | 0.3499 | 0.3068 |
| ES2011a | 102,948 | 9.06% | 0.6407 | 0.5515 |
| ES2011b | 153,281 | 9.95% | 0.7214 | 0.6411 |
| ES2011c | 151,182 | 11.41% | 0.7383 | 0.6485 |
| ES2011d | 187,380 | 8.52% | 0.6941 | 0.5832 |
| IS1008a | 93,671 | 3.24% | 0.4355 | 0.4246 |
| IS1008b | 173,004 | 3.52% | 0.5365 | 0.5100 |
| IS1008c | 144,383 | 6.32% | 0.6386 | 0.5829 |
| IS1008d | 142,855 | 8.70% | 0.6809 | 0.6002 |

### 8.1 Meeting-group comparison

| Meeting group | Mean per-meeting AP | Mean per-meeting overlap F1 | Mean overlap prevalence |
|---|---:|---:|---:|
| ES2003 | 0.2645 | 0.2350 | 3.79% |
| ES2011 | 0.6986 | 0.6061 | 9.74% |
| IS1008 | 0.5729 | 0.5294 | 5.45% |

ES2003 has the lowest overlap AP and F1; ES2011 is strongest, and IS1008 lies between them. These are **descriptive associations**: differences in overlap prevalence, recording conditions, room acoustics, speaker behavior, and annotation quality may contribute. The current evaluation does not establish a causal explanation for the group differences. ES2003 is a useful predeclared difficult group to analyze in M2.

## 9. Relationship to the original smaller baseline

| Metric | Earlier small-data M1 (3 seeds) | Expanded official-group M1 (3 seeds) |
|---|---:|---:|
| Training meetings | 3 | 72 |
| Development overlap AP | 0.3152 ± 0.0176 | 0.5735 ± 0.0080 |
| Development overlap F1 | 0.2290 ± 0.0412 | 0.5187 ± 0.0243 |
| Development macro-F1 | 0.6406 ± 0.0200 | 0.7411 ± 0.0088 |

These values are **not a controlled comparison of training-set size**: the expanded experiment also changes the development meetings. The results indicate better scores under the expanded evaluation protocol, but they cannot isolate the effect of added training data. A controlled dataset-size ablation would require retraining/evaluating both data regimes on a common fixed development set.

## 10. Metrics and evaluation definitions

- **Overlap AP:** Binary average precision for class 2, calculated from softmax class-2 probabilities across all valid development frames; ties handled at common score thresholds. AP summarizes the precision–recall ranking, independently of the argmax operating point.
- **Overlap precision:** `TP_overlap / (TP_overlap + FP_overlap)` based on argmax decisions.
- **Overlap recall:** `TP_overlap / (TP_overlap + FN_overlap)` based on argmax decisions.
- **Overlap F1:** Harmonic mean of overlap precision and recall.
- **Macro-F1:** Unweighted mean of the three per-class F1 scores.
- **Valid frames:** Frames with `valid_mask=True`, excluding uncertain or padded frames. No scoring collar is applied.

The model is selected by pooled development overlap AP, **not** by F1; the highest-F1 epoch need not be the same as the best-AP epoch. No threshold adjustment or test-set selection was carried out for these results.

## 11. Reproducibility

### 11.1 Core implementation files

The full expanded-data scripts should be committed with this report, even if some presently exist only in the local repository. In particular:

```text
src/features/logmel.py
src/models/tcn.py
src/training/engine.py
src/training/evaluation.py
src/training/losses.py                   # XPU-compatible masked loss
src/data/m1_official_dataset.py
scripts/prepare_ami_official_split.py
scripts/generate_frame_labels.py
scripts/batch_generate_ami_labels.py
scripts/audit_ami_annotation_issues.py
scripts/batch_extract_m1_official.py
scripts/compute_m1_official_stats.py
scripts/convert_m1_official_to_memmap.py
scripts/validate_m1_official_loader.py
scripts/train_m1_official_xpu.py
scripts/evaluate_m1_official_development.py
```

### 11.2 Commands

From the repository root in the appropriate Python environment:

```powershell
python -m scripts.validate_m1_official_loader
python -m scripts.train_m1_official_xpu --seed 42 --device xpu --epochs 20 --batch-size 8 --output-dir outputs/training/m1_official/full_xpu_seed42
python -m scripts.train_m1_official_xpu --seed 43 --device xpu --epochs 20 --batch-size 8 --output-dir outputs/training/m1_official/full_xpu_seed43
python -m scripts.train_m1_official_xpu --seed 44 --device xpu --epochs 20 --batch-size 8 --output-dir outputs/training/m1_official/full_xpu_seed44
python -m scripts.evaluate_m1_official_development --device xpu
```

The training commands deliberately use new output folders and the script rejects overwrites of existing experiment artifacts. Device-specific package versions and the exact dataset manifest should be recorded alongside the final repository commit.

### 11.3 Training and evaluation outputs

```text
outputs/dataset_audit/official_es_is/
  train_meetings.txt
  development_meetings.txt
  test_meetings.txt
  frame_label_summary.json
  m1_logmel_train_stats.npz
  m1_feature_extraction_audit.csv
  m1_npy_conversion_audit.csv
  annotation_issues/...
outputs/training/m1_official/
  full_xpu_seed42/{config.json,history.json,best_model.pt,summary.json}
  full_xpu_seed43/{config.json,history.json,best_model.pt,summary.json}
  full_xpu_seed44/{config.json,history.json,best_model.pt,summary.json}
outputs/evaluation/m1_official_development/
  seed_summary.csv
  class_metrics.csv
  confusion_matrices.csv
  per_meeting_metrics.csv
  three_seed_summary.json
  analysis_notes.txt
```

The numerical results in this report were drawn from the corresponding user-provided evaluation and audit exports. Raw AMI recordings and bulky feature arrays are not necessary to include in a Git commit; retaining manifests, scripts, configuration, and compact summaries is usually preferable.

## 12. Limitations

1. **Development-only performance.** The same development set was used to select checkpoints and to report the metrics above. This is not an independent test estimate.
2. **Annotation uncertainty.** The audit detects malformed or missing timestamps and masks some uncertain regions, but does not verify every label against the waveform. Untimed annotations cannot be localized without additional evidence.
3. **Recording heterogeneity.** Large differences among ES2003, ES2011, and IS1008 are descriptive; causes have not been isolated.
4. **Class imbalance and decision rule.** Overlap is relatively rare. AP and argmax-F1 answer different questions, and the current model misses a substantial fraction of overlap frames.
5. **Backend portability.** Intel XPU required a compatible masked-loss implementation and CPU-side metric indexing; the full code and package environment should be preserved for reproducibility.
6. **No threshold optimization or held-out test score.** The reported results use argmax. Any future threshold analysis must be based only on development data and separately labelled, with a predefined protocol before test evaluation.

## 13. Next experimental step: M2

Proceed to M2 with the **same 72/12/12 meeting split, 10 ms labels and masks, 200-frame sequence setup, three seeds, optimizer, training protocol, and pooled development overlap-AP checkpoint selection**. Use the existing log-Mel features and add the planned 2,056 IPD values from four microphone pairs, retaining the previously defined frequency-bin energy mask and feature ordering. Primary M1-versus-M2 comparison: pooled overlap AP, argmax precision/recall/F1 and macro-F1, plus per-meeting AP, particularly for ES2003. Do not choose features, thresholds, or checkpoints using the held-out test recordings.

---

**Conclusion.** The expanded M1 baseline achieves **development overlap AP 0.5735 ± 0.0080**, **overlap F1 0.5187 ± 0.0243**, and **macro-F1 0.7411 ± 0.0088** across three seeds. The main observed limitation is overlap recall (**0.4187 ± 0.0479**), with many overlap frames classified as single-speaker speech. These findings establish the reference against which spatial IPD features in M2 should be evaluated.
