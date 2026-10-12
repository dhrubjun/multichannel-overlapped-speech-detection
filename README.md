# Deep Learning for Multichannel Overlapped Speech Detection

This project studies overlapped speech detection in meeting recordings using deep learning.

The main goal is to investigate whether spatial information from multiple microphones can improve overlapped speech detection compared with using only single-channel speech features.

## Project Goal

The system will classify audio into three classes:

- Silence
- Single-speaker speech
- Overlapping speech

The project will compare different types of audio features and deep-learning models.

## Planned Experiments

### 1. Single-channel baseline
- 80 log-Mel features
- Temporal Convolutional Network (TCN)

### 2. Multichannel model
- Log-Mel features
- Inter-channel Phase Difference (IPD)
- TCN

### 3. Pretrained speech model
- WavLM features
- TCN

### 4. WavLM with spatial features
- WavLM features
- IPD spatial features
- TCN

## Dataset

The project will use the AMI Meeting Corpus.

The main experiments will use far-field multichannel microphone recordings.

The AMI dataset is not included in this repository.

## Evaluation

The main evaluation metrics will include:

- Precision
- Recall
- F1-score
- Average Precision
- Macro-F1
- Confusion matrix

Special attention will be given to the overlapping-speech class.

## Research Questions

1. How well can a deep-learning model distinguish silence, single-speaker speech, and overlapping speech?

2. Does multichannel spatial information improve overlapped speech detection compared with single-channel spectral features?

3. Does spatial information still help when pretrained speech representations such as WavLM are used?

## Current Status

Project setup and dataset validation are currently in progress.
