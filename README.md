# Anomaly Detection and Localization

This repository provides a framework for **Industrial Anomaly Detection and Localization** using state-of-the-art techniques, including **Contrastive Learning** and **Few-shot Similarity Learning**. The project is currently in progress, with efforts focused on validating each method individually before integrating them into a unified framework.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Current State & Future Updates](#current-state--future-updates)

---

## Introduction
<div align="center">
  <img width="639" alt="model_architecture" src="https://github.com/user-attachments/assets/4063a62a-73a4-4938-bbeb-4900de64c373" />
  <p>Model Architecture Diagram</p>
</div>

Traditional anomaly detection methods often suffer from several limitations. One significant drawback is the requirement for a separate model for each dataset or class, which greatly increases complexity and resource demands. Additionally, these approaches fail to capture the relative nature of anomalies, where the definition of "anomalous" can vary depending on the context.

For instance, consider two manufacturing processes producing the same type of product. A characteristic considered "normal" in one process might be regarded as "anomalous" in the other. This context-dependent interpretation of anomalies highlights the limitations of existing methods in adapting to diverse scenarios.

To address these challenges, My ongoing research focuses on developing a unified model that integrates Contrastive Learning and Few-shot Learning. This approach aims to reflect the relative definition of anomalies while improving the efficiency and accuracy of anomaly detection.

## Requirements

Follow the steps below to set up the environment and install dependencies:

1. **Create a Conda virtual environment:**

   ```bash
   conda create -n anomaly-detection python=3.10
   conda activate anomaly-detection
   ```

2. **Install required packages from `requirements.txt`:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Configuration

This project uses [Hydra](https://hydra.cc/) to manage configuration files. The configuration files are organized in the following structure:

```
configs
├── callbacks
│   └── default.yaml
├── data
│   └── default.yaml
├── default.yaml
├── experiment
│   ├── test.yaml  
│   └── default.yaml       
├── key.yaml
└── model
    ├── default.yaml
    ├── resnet.yaml
    └── swin.yaml
```

### 2. Training

Currently, only **Contrastive Learning**-based training is supported.

```bash
python train.py
```

- This implementation reuses ideas from the paper [Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering](https://github.com/HighwayWu/FOCAL), adapting the architecture to **Industrial Anomaly Detection**.
- Key modifications include:
  
  - Using **Swin Transformer** for feature extraction to leverage multi-scale features.
  - Integrating various clustering algorithms and backbone models.

> **Note**: Training supports the validation step but not the testing step yet. Testing support will be added in a future update.

### 3. Testing

Currently, **Few-shot Anomaly Detection** testing is supported using a pretrained model.

```bash
python test.py --dataset "your_data_path"
```

- This implementation is based on [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT) and utilizes:

  - **Swin Transformer** for feature extraction.
  - Patch-level similarity calculations between normal and anomalous images for anomaly localization.

> **Note**: Testing is limited to pretrained models. Training and validation for Few-shot Anomaly Detection will be added in future updates.

---

## Current State & Future Updates

### Current State

1. **Contrastive Learning**:
   - Code adapted from [FOCAL](https://github.com/HighwayWu/FOCAL).
   - Supports training and validation.
   - Test functionality under development.

2. **Few-shot Anomaly Detection**:
   - Code adapted from [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT).
   - Supports testing with pretrained models.
   - Training and validation steps under development.

### Future Updates

- Add testing capabilities for Contrastive Learning.
- Implement training and validation steps for Few-shot Anomaly Detection.
- Integrate Contrastive Learning and Few-shot Similarity Learning into a unified framework.
- Explore advanced feature extraction and clustering techniques.

---

Feel free to contribute or raise issues to improve this repository!
