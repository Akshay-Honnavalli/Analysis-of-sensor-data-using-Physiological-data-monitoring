# 🫁 Oxygen Desaturation Detection from PSG Data Using Minimal Annotations

[![Paper](https://img.shields.io/badge/Published-EAI%20PervasiveHealth-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Detecting oxygen desaturation events in polysomnography (PSG) data using as little as **1–5% labeled data** — matching or beating fully-supervised state-of-the-art performance.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Publication](#publication)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Overview

Oxygen desaturation — a temporary drop in blood oxygen levels — is a key marker of sleep disorders like Obstructive Sleep Apnea (OSA), and repeated episodes are linked to cardiovascular disease, hypertension, and cognitive impairment. Detecting these events from PSG recordings normally requires extensive manual annotation from sleep specialists, which is slow and expensive.

This project explores how far detection accuracy can be pushed with **minimal expert annotation**, combining unsupervised anomaly detection with lightly-supervised classification on partially labeled data.

## Dataset

- **[NCH Sleep DataBank](https://physionet.org/)** (Nationwide Children's Hospital) — 3,984 pediatric sleep studies across 3,673 patients
- 28 channels: EEG, ECG, EOG, EMG, SpO2, respiratory airflow, etc., mostly sampled at 256 Hz
- Highly imbalanced (desaturation vs. non-events ranging from 1:99 to 10:90) and temporally skewed
- Sleep stages labeled per AASM v2.1 guidelines (Wake, N1, N2, N3, REM)

> ⚠️ The NCH Sleep DataBank requires a data use agreement via PhysioNet — it is not bundled in this repo.

## Approach

**1. Preprocessing**
- Per-sleep-stage segmentation (W, N1, N2, N3, REM analyzed separately)
- Butterworth bandstop filtering (60 Hz + 120 Hz) for EEG denoising, applied zero-phase
- Simple Moving Average smoothing for non-EEG channels
- Robust Scaling to handle physiological signal outliers

**2. Unsupervised anomaly detection (baseline)**
- Random Forest Gini Index used to rank channel importance — confirmed SpO2 and respiratory channels as most predictive
- LSTM Autoencoder (2 encoding + 2 decoding layers, bottleneck) trained per sleep stage, channels added incrementally by importance until AUC began to decline
- Reconstruction error used as the anomaly signal

**3. Supervised detection on partially annotated data (core contribution)**
- Random Forest Classifier trained on as little as **1–5%** annotated data
- Compared random splits (1:99, 5:95) vs. systematic splits (1-in-20, 1-in-33, 1-in-100 rows) to mimic how a specialist naturally annotates a contiguous block
- Benchmarked against frequency-domain (FFT) and time-frequency-domain (Wavelet Transform) models trained on 100% annotations

## Results

| Method | Annotation Used | Balanced Accuracy |
|---|---|---|
| LSTM Autoencoder (unsupervised) | 0% | ~0.71 (weighted avg) |
| Prior state-of-the-art (EEG-based) | 100% | 0.58 – 0.71 |
| RFC, systematic 1-in-20 sampling | **5%** | **0.98 – 0.99** |
| RFC, random 5:95 split | 5% | 0.96 – 0.99 |
| RFC, systematic 1-in-100 | 1% | 0.85 – 0.92 |

- **>0.96 balanced accuracy using only 5% labeled data** — exceeding prior state-of-the-art trained on fully annotated EEG data
- Systematic sampling consistently outperformed random sampling at equivalent label budgets
- Time-domain RFC on partial labels outperformed frequency- and time-frequency-domain models trained on full annotations

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Usage

```bash
# Convert raw EDF recordings to CSV
python preprocess.py --input data/raw --output data/processed

# Train the Random Forest Classifier on partially annotated data
python train_rfc.py --annotation-pct 5 --split systematic

# Train the LSTM Autoencoder baseline
python train_autoencoder.py --sleep-stage N2
```

> Update these commands/paths to match your actual script names before publishing.

## Tech Stack

- Python, NumPy, pandas, SciPy
- scikit-learn (Random Forest, Gini feature importance)
- TensorFlow / Keras (LSTM Autoencoder)
- Signal processing: Butterworth filtering, FFT, Wavelet Transform

## Publication

Published at **EAI PervasiveHealth**.
Authors: Akshay Honnavalli, Hrishi Preetham G L, Aaryan Hemant Badyal, Adithya Sharma, Gowri Srinivasa — PES Center for Pattern Recognition, PES University, Bengaluru.

## Future Work

- Fully automated detection pipeline requiring zero manual annotation
- Generalizability testing across different age groups beyond the pediatric NCH cohort

## Citation

```bibtex
@inproceedings{honnavalli2025oxygen,
  title={Detection of Oxygen Desaturation Events from PSG Data Using Minimal Annotations},
  author={Honnavalli, Akshay and G L, Hrishi Preetham and Badyal, Aaryan Hemant and Sharma, Adithya and Srinivasa, Gowri},
  booktitle={EAI PervasiveHealth},
  year={2025}
}
```

---

<p align="center"><i>Built with the goal of reducing clinician workload in sleep disorder diagnostics.</i></p>
