# M2-D2: Microphone Geometry and IPD Phase Analysis

## 1. Objective

Investigate whether the microphone-pair selection, phase extraction, and spatial properties of AMI Array 1 recordings could explain the limited performance of the original M2 model.

The original M2 combines 80 log-Mel features with 2,056 IPD features from four microphone pairs. On the ES2003a development meeting, the weighted M2 model achieved an overlap average precision (AP) of 0.130347, compared with 0.166336 for the weighted single-channel M1 baseline.

This investigation analyzes the spatial inputs without modifying or retraining either model.

## 2. Experimental Configuration

### Dataset

AMI Array 1 recordings from the training meetings:

- ES2008a
- ES2008b
- ES2008c

The development meeting ES2003a and held-out test meeting ES2004a were not used for these spatial diagnostics.

### Original IPD configuration

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz |
| STFT window | 400 samples (25 ms) |
| Hop | 160 samples (10 ms) |
| FFT length | 512 |
| Window | Periodic Hann |
| Microphone pairs | (01,05), (02,06), (03,07), (04,08) |
| Representation | Cosine and sine of cross-spectral phase |
| Mask threshold | -30 dB |

The cross-spectrum is defined as:

\[
C_{ij}(t,f)=X_i(t,f)X_j^*(t,f)
\]

The IPD representation is:

\[
[\cos(\angle C_{ij}),\sin(\angle C_{ij})]
\]

The documentation describes AMI Array 1 as an eight-microphone circular array with a radius of approximately 10 cm. The exact channel-to-position mapping was not conclusively verified during this investigation.

## 3. Time-Delay Analysis

Twelve two-second annotated single-speaker windows were selected across ES2008a, from approximately 32.68 to 1036.31 seconds.

All 28 unique microphone pairs were evaluated, resulting in 336 delay estimates.

For an assumed maximum array diameter of 20 cm, the maximum direct-path acoustic delay at 16 kHz is approximately 9.33 samples, using a sound speed of 343 m/s.

### Results

- Total windows: 12
- Total pairwise estimates: 336
- Estimates outside the expected physical bound: 0
- Median absolute closure error for the tested microphone triplet: 0 samples
- Maximum absolute closure error: 2 samples

The delay measurements are physically plausible and generally consistent with the assumed array dimensions. However, cross-correlation delays do not independently establish microphone positions or direct-path propagation.

## 4. Frequency-Dependent Coherence

Magnitude-squared coherence was analyzed for all 28 microphone pairs.

The resulting values were grouped by circular channel-index separation, assuming cyclic numbering for the grouping analysis.

| Band (Hz) | Separation 1 | Separation 2 | Separation 3 | Separation 4 |
|---|---:|---:|---:|---:|
| 0–500 | 0.9323 | 0.8381 | 0.7654 | 0.7396 |
| 500–1000 | 0.7321 | 0.4033 | 0.2685 | 0.2394 |
| 1000–2000 | 0.3947 | 0.2689 | 0.2569 | 0.2442 |
| 2000–4000 | 0.2406 | 0.2146 | 0.2182 | 0.2167 |
| 4000–8000 | 0.1844 | 0.1689 | 0.1646 | 0.1670 |

Adjacent-index pairs exhibit substantially higher coherence than the original M2 pairs at lower frequencies, particularly from 500 to 1000 Hz.

This pattern is compatible with consecutive microphone numbering around the circular array but does not constitute independent confirmation of the physical mapping.

Higher coherence does not necessarily imply better speaker-count discrimination. Alternative microphone pairs require controlled evaluation before any replacement is justified.

## 5. Phase Consistency

The measured cross-spectral phase was compared with the phase predicted by time-domain delay estimates.

For the original four microphone pairs:

| Band (Hz) | Mean coherence | Mean absolute phase residual |
|---|---:|---:|
| 0–500 | 0.7393 | 9.54° |
| 500–1000 | 0.2323 | 15.44° |
| 1000–2000 | 0.2350 | 27.97° |
| 2000–4000 | 0.2034 | 53.99° |
| 4000–8000 | 0.1923 | 67.11° |

Phase agreement is strongest below 500 Hz. At higher frequencies, coherence generally decreases and the residual relative to a constant-delay model increases.

These measurements are influenced by room reverberation, acoustic motion, frequency-dependent signal energy and phase ambiguity. They do not independently establish that high-frequency IPD should be discarded.

## 6. Synthetic IPD Validation

Two synthetic experiments were conducted.

### Broadband delay test

A broadband signal was delayed by four samples.

| Measurement | Result |
|---|---:|
| Expected delay | 4 samples |
| Estimated delay | 4 samples |
| Cross-correlation | 1.000000 |
| Mean phase residual | 2.0719° |
| Median phase residual | 1.2091° |
| 95th-percentile residual | 6.3829° |

The time-domain delay was recovered correctly. The finite-window STFT introduced nonzero broadband phase residuals.

### Sinusoidal phase test

A 1000 Hz sinusoid was delayed by four samples at 16 kHz.

| Measurement | Result |
|---|---:|
| Expected phase | -90.0000° |
| Measured phase | -90.0000° |
| Mean absolute phase error | 0.000000° at displayed precision |

The sinusoidal experiment supports the correctness of the IPD sign convention and its representation of a known phase difference.

## 7. Single-Speaker Versus Overlap Analysis

A total of 120 annotated 300 ms windows were selected from the three training meetings:

- 60 single-speaker windows
- 60 overlap windows

The analysis used real surrounding audio context to reproduce the original M2 STFT frame geometry. Synthetic equivalence testing against `extract_ipd()` passed.

The four original M2 pairs were compared with eight adjacent-index pairs, generating 7,200 window-pair-frequency-band records.

### Coherence differences

The results did not show that overlap consistently produces lower coherence.

For example, in the 2000–4000 Hz band, the overlap-minus-single-speaker coherence differences were positive for all three meetings:

| Meeting | Original pairs | Adjacent pairs |
|---|---:|---:|
| ES2008a | +0.0490 | +0.0510 |
| ES2008b | +0.0726 | +0.0791 |
| ES2008c | +0.0240 | +0.0290 |

These differences are descriptive and may be affected by speaker position, signal energy, recording conditions and sampling variability.

### Mask retention

The existing -30 dB mask retains progressively fewer IPD values at higher frequencies.

| Band (Hz) | Single-speaker retention | Overlap retention |
|---|---:|---:|
| 0–500 | 96.29% | 97.09% |
| 500–1000 | 78.43% | 78.80% |
| 1000–2000 | 52.20% | 51.95% |
| 2000–4000 | 17.49% | 17.29% |
| 4000–8000 | 7.76% | 6.28% |

For the original M2 pairs, only 14.57% of the frequency bins in the 4000–8000 Hz band had sufficient retained frames to calculate phase concentration in the sampled overlap windows.

This limits the interpretation of high-frequency phase-concentration statistics and motivates a controlled masking ablation.

## 8. Main Findings

1. Synthetic delay and phase tests support the mathematical correctness of the original IPD extractor.
2. The estimated microphone-pair delays are physically plausible.
3. Coherence and phase consistency vary substantially with frequency and microphone-pair selection.
4. Adjacent-index pairs exhibit higher low-frequency coherence than the original M2 pairs.
5. Overlapping speech does not consistently correspond to lower coherence or phase concentration.
6. The -30 dB mask removes a large proportion of higher-frequency IPD values.
7. The current diagnostic evidence does not establish which microphone-pair configuration yields better overlap detection.

## 9. Limitations

- Exact physical channel ordering remains unverified.
- Only three training meetings were analyzed.
- Speech-state comparisons used 20 selected windows per class per meeting.
- Descriptive differences were not evaluated using independent, episode-level statistical inference.
- Coherence and phase concentration are diagnostic measures rather than direct classification metrics.
- No alternative microphone-pair configuration was trained or evaluated.
- No masking-threshold ablation was conducted in this issue.

## 10. Conclusion and Next Steps

The original IPD implementation passed synthetic correctness checks, and the AMI recordings contain measurable frequency-dependent spatial relationships.

The analysis identifies microphone-pair selection, high-frequency phase behaviour and energy masking as plausible areas for further investigation. None has yet been demonstrated to cause the lower development overlap AP of M2.

The next experiment, M2-D3, will compare unmasked IPD with -20 dB, -30 dB and -40 dB masking under an otherwise controlled training protocol.

Subsequent experiments will examine compact IPD representations and separate spatial-feature encoders.

## 11. Reproducibility

Relevant scripts:

- `scripts/analyze_m2_channel_phase.py`
- `scripts/audit_m2_single_speaker_delays.py`
- `scripts/analyze_m2_coherence.py`
- `scripts/analyze_m2_all_pairs_coherence.py`
- `scripts/analyze_m2_phase_consistency.py`
- `scripts/validate_m2_ipd_sign.py`
- `scripts/compare_m2_speech_states.py`

Generated results are stored under:

`outputs/evaluation/m2_channel_phase/`

Large generated outputs are not required in Git if their generation procedure and experiment settings are preserved.