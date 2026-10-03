# GA-ANFIS EEG Seizure Detection

A seizure detection system using EEG signal features, Genetic Algorithm
feature selection, and an Adaptive Neuro-Fuzzy Inference System (ANFIS).

## Pipeline

CHB-MIT EEG
    ↓
Preprocessing
    ↓
5-second window segmentation
    ↓
EEG feature extraction
    ↓
Genetic Algorithm feature selection
    ↓
107 selected features
    ↓
ANFIS classifier
    ↓
Probability thresholding
    ↓
Event-level post-processing
    ↓
Seizure / Normal

## Dataset

Dataset: CHB-MIT EEG

Experiments used:
- 7 EDF recordings
- 5033 EEG windows
- 4988 normal windows
- 45 seizure windows

## Features

Extracted features include:

- Mean
- Standard deviation
- Variance
- RMS
- Energy
- Delta power
- Theta power
- Alpha power
- Beta power
- Gamma power

## Feature Selection

The Genetic Algorithm selected 107 features from the
original feature set.

## ANFIS

The selected features are provided to an ANFIS
neuro-fuzzy classifier for seizure classification.

Final event configuration:

- Probability threshold: 0.45
- Merge gap: 30 seconds
- Minimum event duration: 10 seconds

## Consolidated Results

| Model / operating point | Window precision | Window recall | Window F1 | Window false positives / false negatives | Event detection | Event false alarms |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ML | 76.0% | 42.2% | 0.5429 | Not reported here | 100% (4/4) | 0 |
| Class-weighted ML @ 0.50 | 53.8% | 77.8% | 0.636 | 6 / 2 | — | — |
| Class-weighted ML @ 0.25* | 42.1% | 88.9% | 0.571 | 11 / 1 | — | — |
| GA + ANFIS @ 0.45 | Not reported here | Not reported here | 0.7059† | Not reported here | 100% (4/4) | 3 |

\* The 0.25 threshold was selected using validation data. Both class-weighted ML operating points are reported on the same held-out test set; the false-positive and false-negative counts are **windows**, not seizure events.

† The ANFIS window-level F1 comes from an earlier evaluation setup. It is included for context and should not be compared directly with the class-weighted ML test results. ANFIS event results are a separate event-level evaluation. Event evaluation contains only four seizures, so these results are experimental rather than evidence of clinical performance.

These results show a recall improvement over the earlier baseline window recall of 42.2%. No model or threshold change is implied by this summary.

## Limitations

The event-level evaluation contains only four seizure events.
Therefore, the results should be interpreted as an experimental
evaluation rather than evidence of clinical performance.

## Future Work

- Evaluate on additional patients
- Patient-independent validation
- Improve temporal localization
- Reduce false alarms
- Explore additional EEG features
- Real-time EEG inference
