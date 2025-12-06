# WavLM Fine-Tuning for Speech Emotion Recognition (SER)

This repository contains the training code used in the accompanying article on cross-lingual zero-shot speech emotion recognition for Polish.  
The system was developed for the **PolEval 2025 SER Challenge**, where models must be trained solely on multilingual data and evaluated on Polish speech.

## Description

The script fine-tunes **WavLM (microsoft/wavlm-base-plus)** on the multilingual **CAMEO** dataset.  
It loads file lists from the PolEval repository, downloads audio from Hugging Face, processes audio with `Wav2Vec2FeatureExtractor`, and trains a 6-class emotion classifier.

## Data Setup

Download the PolEval SER data from:

https://github.com/poleval/2025-speech-emotion

Required files:

```
train/in.tsv
dev/in.tsv
```

Set the dataset path in the script:

```python
BASE_PATH = "PATH_TO_SER_DATA"
```

Example:

```python
BASE_PATH = "/home/user/2025-speech-emotion"
```

## Running

Install dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas
```

Run training:

```bash
python train_wavlm.py
```

The trained model will be saved to:

```
./outputs/wavlm_finetune/final_model
```