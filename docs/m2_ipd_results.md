# M2: Multichannel IPD Feature Extraction

## 1. Objective

Investigate whether inter-channel phase difference (IPD) features improve overlapping-speech detection compared with the M1 single-channel log-Mel baseline.

## 2. Feature Configuration

The AMI Array 1 recordings contain eight microphones arranged in a circular array with an approximately 10 cm radius.

Four fixed microphone pairs were selected for the initial M2 model:

- Array1-01 and Array1-05
- Array1-02 and Array1-06
- Array1-03 and Array1-07
- Array1-04 and Array1-08

The physical microphone numbering has not been independently verified. Therefore, these combinations are treated as candidate channel pairs rather than confirmed opposite-microphone pairs.

### STFT and IPD Parameters

| Parameter | Configuration |
|---|---|
| Dataset | AMI Meeting Corpus, Array 1 |
| Microphones | Eight |
| Sampling rate | 16 kHz |
| STFT window | 25 ms, periodic Hann |
| Frame hop | 10 ms |
| FFT size | 512 |
| Frequency bins | 257 |
| IPD representation | Cosine and sine |
| Energy mask | -30 dB relative to the maximum cross-spectrum magnitude within each frame |
| Masked bins | Both phase components set to zero |
| IPD dimensions | 2,056 per frame |

The framing configuration matches the existing M1 log-Mel implementation.

## 3. Multichannel Audio and Synchronization Validation

All eight microphone recordings were validated for the following meetings:

- ES2008a
- ES2008b
- ES2008c
- ES2003a

Each meeting passed the checks for matching sample rates, sample counts, channel counts and sample widths.

Cross-correlation analysis was performed at five recording positions for each of the four selected microphone pairs, producing 80 measurements in total.

The measured relative delays ranged from -0.5625 ms to +0.5625 ms.

These delays are physically plausible for the assumed microphone-array dimensions. However, cross-correlation includes acoustic propagation effects and cannot independently establish sample-perfect synchronization or exclude small recording offsets.

No artificial time alignment was applied. The original microphone sample indices were preserved to retain the spatial arrival-time information required for IPD extraction.

## 4. IPD Implementation and Validation

IPD was calculated from the complex STFT cross-spectrum:

\[
C_{ij}(t,f)=X_i(t,f)X_j^*(t,f)
\]

The phase representation uses:

\[
\cos(\mathrm{IPD}_{ij})=\frac{\Re(C_{ij})}{|C_{ij}|}
\]

\[
\sin(\mathrm{IPD}_{ij})=\frac{\Im(C_{ij})}{|C_{ij}|}
\]

Negligible-energy bins and bins suppressed by the optional energy mask are represented by zero-valued cosine and sine components.

Synthetic tests passed for feature dimensions, timestamp alignment, identical microphone signals, known sample delay, silence and invalid inputs.

The optional energy-masking implementation also passed validation.

### Real-Audio Validation

A five-second segment from ES2008a was used.

| Property | Result |
|---|---|
| Audio shape | (8, 80000) |
| Log-Mel shape | (500, 80) |
| IPD shape | (500, 4, 257, 2) |
| Timestamp alignment | PASS |
| Finite output | PASS |

All 500 corresponding label frames were valid.

## 5. Noise-Sensitivity Analysis

Independent Gaussian noise was added to each microphone recording, with noise RMS equal to 1% of the overall audio RMS.

Across four microphone pairs, the average of the reported median phase changes was:

| Energy selection | Median phase change, averaged across pairs |
|---|---:|
| Lowest 20% | 78.91° |
| Highest 20% | 0.63° |

Weak-energy frequency bins were substantially more sensitive to the added noise than high-energy bins.

These results measure noise-induced phase changes, not absolute phase estimation accuracy.

## 6. Energy Threshold Comparison

Relative energy masking was evaluated using thresholds of -20, -30 and -40 dB.

| Threshold | Mean bins retained | Mean median phase change |
|---|---:|---:|
| No mask | 100.00% | 7.348° |
| -40 dB | 55.17% | 2.563° |
| -30 dB | 28.28% | 1.117° |
| -20 dB | 10.49% | 0.379° |

The -30 dB threshold was selected as an initial compromise between retaining frequency bins and reducing sensitivity to injected noise.

This threshold has not been optimized or validated for overlapped-speech classification performance.

The energy mask retains the original frequency-bin positions, setting suppressed IPD components to zero. Consequently, masking does not reduce the dimensionality of the dense feature representation.

## 7. Class-Wise Mask Analysis

Sixty five-second segments were sampled from training meeting ES2008a. The selection intentionally included overlap-rich segments.

Only frames accepted by the existing annotation valid mask were included.

| Speech class | Valid frames | Mean bins retained |
|---|---:|---:|
| No annotated speech | 3,730 | 22.35% |
| Single speaker | 23,279 | 25.82% |
| Overlap | 2,579 | 26.36% |

Retention percentages are averaged across four microphone pairs.

The -30 dB mask did not disproportionately suppress overlapping-speech frames in this sampled data.

Because overlap-rich segments were deliberately selected, these results should not be interpreted as representative class proportions for the entire training dataset.

## 8. Initial M2 Model Input

The initial M2 input will combine:

- 80 log-Mel features from Array1-01.
- 2,056 IPD features from four microphone pairs.

The total input dimensionality is:

\[
80+(4\times257\times2)=2136
\]

The existing M1 TCN temporal architecture will be preserved, while its input projection will be adapted to accept the combined features.

## 9. Limitations

- Only selected short audio segments were used for detailed IPD experiments.
- Real-audio phase ground truth is unavailable.
- Physical microphone channel ordering remains unverified.
- Cross-correlation measurements do not prove sample-perfect synchronization.
- Greater phase stability does not guarantee improved classification performance.
- Independently extracted five-second clips use local boundary padding.
- Energy masking does not reduce the dense input dimensionality.
- The -30 dB threshold has not been optimized for overlap detection.
- The held-out test set was not used for feature selection or parameter tuning.

## 10. Next Steps

- Retain the selected pairs as fixed candidate channel combinations for the initial M2 experiment.
- Implement combined log-Mel and IPD feature extraction under M2.3.
- Validate full-recording feature and label alignment.
- Implement memory-efficient feature loading.
- Compute required normalization statistics using training data only.
- Adapt the TCN input projection and train M2.
- Compare M2 against the unchanged M1 baseline using the development set.

The physical microphone mapping should be revisited if authoritative channel-position information becomes available.