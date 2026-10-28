\# M1: Single-Channel Log-Mel + TCN — Baseline Evaluation



\## 1. Objective



The objective of M1 was to develop and evaluate a single-channel deep learning baseline for overlapped speech detection using the AMI Meeting Corpus.



The model classifies every 10 ms audio frame into three states:



\- \*\*Class 0:\*\* No annotated speech

\- \*\*Class 1:\*\* Single-speaker speech

\- \*\*Class 2:\*\* Overlapping speech (two or more simultaneous speakers)



This baseline provides a reference for evaluating the contribution of multichannel spatial features in M2.



\## 2. Dataset



The selected input was microphone Array1-01 from the AMI Meeting Corpus, sampled at 16 kHz.



| Split | Meetings | Valid frames |

|---|---|---:|

| Training | ES2008a, ES2008b, ES2008c | 525,629 |

| Development | ES2003a | 111,472 |

| Test | ES2004a | 102,016 |



The test meeting was not used for training, model selection, or development analysis.



\### Training class distribution



| Class | Frames | Percentage |

|---|---:|---:|

| No annotated speech | 108,809 | 20.70% |

| Single speaker | 385,290 | 73.30% |

| Overlap | 31,530 | 6.00% |



The low frequency of overlapping speech is an important challenge.



\## 3. Feature Extraction



The input features were extracted using:



\- A 25 ms Hann analysis window

\- A 10 ms frame hop

\- A 512-point FFT

\- 80 log-Mel spectral features

\- Normalization using training-only statistics



Training sequences contained 200 frames (2 seconds), with a stride of 100 frames. Development sequences used a stride of 200 frames to avoid duplicate frame-level evaluation.



\## 4. TCN Architecture



A noncausal residual dilated Temporal Convolutional Network (TCN) was implemented.



| Component | Configuration |

|---|---|

| Input | 80 log-Mel features |

| Hidden channels | 64 |

| Residual blocks | 5 |

| Convolutions per block | 2 |

| Kernel size | 3 |

| Dilations | 1, 2, 4, 8, 16 |

| Activation | ReLU |

| Dropout | 0.1 |

| Receptive field | 125 frames |

| Output | Three class logits per frame |

| Trainable parameters | 128,899 |



The network uses an initial 1×1 convolution to project 80 input features into 64 hidden channels, followed by five residual blocks and a final 1×1 three-class classification layer.



\## 5. Training Configuration



\- Optimizer: AdamW

\- Learning rate: 0.001

\- Weight decay: 0.0001

\- Batch size: 8

\- Maximum epochs: 20

\- Gradient clipping: 1.0

\- Early-stopping patience: 5

\- Checkpoint selection: Highest development overlap AP

\- Random seed: 42

\- Invalid labels and padded frames excluded from training loss



Two training objectives were investigated:



1\. Unweighted cross-entropy

2\. Square-root inverse-frequency weighted cross-entropy



The weighted configuration used class weights of approximately `\[1.0000, 0.5314, 1.8577]`.



\## 6. Development Results



Both models were evaluated on the complete ES2003a development meeting.



Results are reported at the checkpoints selected by development overlap AP.



| Metric | Unweighted | Sqrt-weighted |

|---|---:|---:|

| Selected epoch | 5 | 3 |

| Overall accuracy | 85.97% | 82.15% |

| Macro-F1 | 0.6397 | 0.6253 |

| Overlap precision | 0.1918 | 0.1175 |

| Overlap recall | 0.1601 | 0.3548 |

| Overlap F1 | 0.1745 | 0.1765 |

| Overlap AP | 0.1470 | 0.1663 |



\### Class-wise recall



| Class | Unweighted | Sqrt-weighted |

|---|---:|---:|

| No annotated speech | 93.01% | 95.59% |

| Single speaker | 83.35% | 74.12% |

| Overlap | 16.01% | 35.48% |



Class weighting improved overlap AP by approximately 13.1% relative to the unweighted baseline, and substantially increased overlap recall.



However, overlap precision decreased because the weighted model generated more false overlap detections.



\## 7. Confusion Matrices



Rows represent reference labels and columns represent predictions.



\### Unweighted TCN



| Actual / Predicted | No speech | Single | Overlap |

|---|---:|---:|---:|

| No speech | 42,485 | 3,174 | 20 |

| Single | 9,110 | 52,989 | 1,476 |

| Overlap | 187 | 1,676 | 355 |



\### Sqrt-weighted TCN



| Actual / Predicted | No speech | Single | Overlap |

|---|---:|---:|---:|

| No speech | 43,663 | 1,899 | 117 |

| Single | 10,657 | 47,124 | 5,794 |

| Overlap | 242 | 1,189 | 787 |



The dominant error was confusion between single-speaker speech and overlapping speech.



\## 8. Post-Processing Experiments



Probability-threshold adjustment and minimum-duration filtering were investigated on the development meeting.



For the weighted model:



| Method | Precision | Recall | Overlap F1 |

|---|---:|---:|---:|

| Three-class argmax | 0.1175 | 0.3548 | 0.1765 |

| Binary overlap threshold 0.75 | 0.2017 | 0.2083 | 0.2050 |

| Threshold 0.65 + 50 ms minimum duration | 0.1841 | 0.2376 | 0.2074 |



These development-tuned settings improved the overlap F1 operating point modestly but did not substantially resolve the precision-recall tradeoff. Binary threshold results are not complete three-class classification metrics.



\## 9. False-Positive Analysis



Long false overlap predictions were identified and compared with reference labels.



Three representative regions were:



\- 570.53–571.51 s

\- 742.00–742.89 s

\- 826.27–827.15 s



All 275 frames across these intervals were labeled as valid single-speaker speech.



The model also made false overlap predictions lasting several hundred milliseconds. Consequently, removing only short predicted overlap segments was insufficient.



An analysis of predictions near 2-second sequence boundaries found no substantial increase in false-overlap rates at those boundaries.



The reference labels establish that these are false positives under the adopted annotation protocol; an independent audit of possible background voices has not been completed.



\## 10. Discussion



The TCN learned useful spectral and temporal representations for speech-state classification.



It performed relatively well for no-speech and single-speaker frames but had difficulty identifying overlapping speech.



Class weighting improved overlap sensitivity but caused additional false-positive detections. Threshold adjustment provided a different precision-recall operating point rather than fundamentally improving the model's separation ability.



Further work should investigate whether spatial information can improve overlap discrimination.



\## 11. Limitations



\- Only one microphone was used.

\- The system does not explicitly model spatial differences between sound sources.

\- The model is noncausal and uses fixed-length sequences.

\- Results come from one development meeting and one random seed.

\- The held-out test meeting remains unevaluated.

\- Threshold and duration parameters were explored on development data and may overfit that meeting.



\## 12. Next Steps



M2 will extend the baseline with multichannel Inter-channel Phase Difference (IPD) features.



The same TCN design, data splits, and evaluation metrics will be retained where possible to isolate the contribution of spatial information.



The main research question is whether spatial features improve overlapping-speech detection compared with the single-channel spectral baseline.

