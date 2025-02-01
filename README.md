# ECG Wavelet Analysis and Subject Identification

This repository contains a Python script for analyzing ECG signals using wavelet transforms and machine learning techniques. The script processes ECG data, extracts features, and uses Random Forest and SVM classifiers to identify subjects based on their ECG signals. The dataset used in this project is the **PTB Diagnostic ECG Database**, which can be accessed from [PhysioNet](https://www.physionet.org/content/ptbdb/1.0.0/).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)

## Overview

The goal of this project is to analyze ECG signals using wavelet transforms and machine learning models to identify subjects based on their ECG data. The script processes raw ECG signals, extracts features using wavelet decomposition and autocorrelation, and trains Random Forest and SVM classifiers for subject identification.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scipy
- Scikit-learn
- PyWavelets (pywt)
- WFDB (Waveform Database) library
- Statsmodels

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecg-wavelet-analysis.git
   cd ecg-wavelet-analysis
