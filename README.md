# OriENCODE: [Mitigating Out-of-Focus Noises in Single-Molecule Localization via Orientation-aware Deep Network]

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Paper: Under Review](https://img.shields.io/badge/Paper-Under_Review-yellow.svg)](#)


This repository contains the official implementation for the paper: **"Mitigating Out-of-Focus Noises in Single-Molecule Localization via Orientation-aware Deep Network"**.

## 📖 Introduction
Single-molecule localization microscopy (SMLM) often suffers from performance degradation in thick perinuclear regions due to severe structured noise from out-of-focus fluorescence and overlapping point spread functions (PSFs). These issues obscure faint in-focus emitters and bias localization estimates, preventing conventional methods from reaching theoretical precision limits. 

To address this challenge, we present **ORIENCODE (ORIENtation-ENCODEd)**, a robust, physically interpretable deep-learning framework designed for single-molecule localization under complex out-of-focus interference.

**Key Features and Contributions:**
* **Geometric Perception via Euler's Elastica Energy:** ORIENCODE leverages an Euler's elastica energy model to jointly constrain boundary length and curvature. This allows the network to robustly distinguish in-focus emitters (compact, uniform curvature) from out-of-focus diffuse backgrounds.
* **CRLB-Anchored Optimization:** The model integrates an orientation-aware neural network module with adversarial training and a Cramér-Rao lower bound (CRLB)-anchored loss function to minimize uncertainty and approach the theoretical localization limit.
* **Adaptive Detection Thresholding:** A lightweight post-processing CNN is employed to adaptively determine pixel-level detection thresholds based on local molecular density.

## ⚙️ Installation
The code has been tested with Python 3.8.17 and PyTorch 1.7.1. Please follow the steps below to set up the environment:

```bash
# Clone the repository
git clone https://github.com/GaoQF1016/OriENCODE.git
cd OriENCODE

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate oriencode_env

# Alternatively, using pip:
# pip install -r requirements.txt
