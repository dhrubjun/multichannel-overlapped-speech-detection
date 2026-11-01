
# M2: Multichannel IPD Feature Extraction

## 1. Objective

Investigate whether inter-channel phase difference (IPD)
features can improve overlapping-speech detection compared
with the M1 single-channel log-Mel baseline.

## 2. Feature Configuration

- Dataset: AMI Meeting Corpus, Array 1
- Microphones: Eight
- Candidate pairs: 01-05, 02-06, 03-07, 04-08
- Sampling rate: 16 kHz
- STFT window: 25 ms, periodic Hann
- Hop: 10 ms
- FFT size: 512
- Frequency bins: 257
- Representation: cosine and sine of IPD
- Energy mask: -30 dB relative to each frame's
  maximum cross-spectrum magnitude
- Masked bins: Both phase components set to zero
- Output: 2056 IPD features per frame

The physical microphone numbering still requires
verification against the AMI array geometry.

## 3. Validation

Synthetic tests passed for:
- Feature shape and timestamps
- Identical microphone signals
- Known sample delay
- Silence handling
- Alignment with M1 log-Mel
- Invalid inputs

The optional energy mask also passed validation.

Real-audio validation on a five-second ES2008a segment:
- Audio shape: (8, 80000)
- Log-Mel shape: (500, 80)
- IPD shape: (500, 4, 257, 2)
- Timestamp alignment: PASS
- Finite output: PASS

## 4. Noise-Sensitivity Analysis

Independent Gaussian noise was added to each microphone,
with RMS equal to 1% of the overall audio RMS.

Across four microphone pairs, the average of the reported
median phase changes was:

| Energy selection | Phase change |
|---|---:|
| Lowest 20% | 78.91 degrees |
| Highest 20% | 0.63 degrees |

Weak-energy bins were substantially more sensitive to
the added noise.

These values measure noise-induced phase changes, not
absolute phase estimation accuracy.

## 5. Energy Threshold Comparison

Results for a five-second ES2008a segment:

| Threshold | Mean bins retained | Mean median phase change |
|---|---:|---:|
| No mask | 100.00% | 7.348 degrees |
| -40 dB | 55.17% | 2.563 degrees |
| -30 dB | 28.28% | 1.117 degrees |
| -20 dB | 10.49% | 0.379 degrees |

A -30 dB threshold was selected as an initial compromise
between retaining frequency bins and reducing sensitivity
to the injected noise.

This threshold has not been validated for classification
performance.

## 6. Class-Wise Mask Analysis

Sixty five-second segments were selected from training
meeting ES2008a. The selection intentionally included
overlap-rich segments.

Only frames accepted by the annotation valid mask
were included.

| Class | Frames | Mean bins retained |
|---|---:|---:|
| No annotated speech | 3730 | 22.35% |
| Single speaker | 23279 | 25.82% |
| Overlap | 2579 | 26.36% |

Values are averaged across four microphone pairs.

The -30 dB mask did not disproportionately suppress
overlap frames in this sample.

## 7. Initial M2 Input

- Log-Mel: 80 dimensions
- IPD: 4 pairs x 257 bins x 2 components
- Total: 2136 dimensions per frame

The M1 TCN temporal architecture will be retained.
Only the input projection will need adaptation.

## 8. Limitations

- Only selected short audio segments were analyzed.
- Real-audio phase ground truth is unavailable.
- Physical microphone channel ordering remains unverified.
- Phase stability does not guarantee classification gains.
- Five-second clips use local boundary padding.
- Masking does not reduce dense feature dimensionality.
- No held-out test audio was used for parameter selection.

## 9. Next Steps

- Verify microphone numbering and geometry under M2.1.
- Finalize IPD representation after any required pair changes.
- Implement combined log-Mel and IPD features under M2.3.
- Compute training-only normalization statistics.
- Compare M2 with the unchanged M1 baseline.