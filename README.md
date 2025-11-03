# Botnet Traffic Classification

A comprehensive machine learning pipeline for detecting and classifying botnet traffic in network flow data. This project utilizes the CTU-13 dataset to evaluate detection performance across three distinct experimental scenarios: single-dataset baseline analysis, cross-dataset generalization, and multi-class botnet identification.

## Overview

This project implements a supervised learning approach to analyze network traffic flows. It is structured into three experimental modules, each designed to address a specific aspect of network intrusion detection:

1.  **EXP1 (Baseline):** Internal consistency and performance assessment on individual datasets.
2.  **EXP2 (Generalization):** Model benchmarking (RF, DT, MLP, SVM) and transfer learning capabilities across different network environments.
3.  **EXP3 (Multi-Class):** Granular classification distinguishing between specific botnet families and background traffic.

## Features

- **Multi-Model Support:** Implementations of Random Forest, Decision Tree, Multi-Layer Perceptron (MLP), and Support Vector Machines (SVM).
- **Advanced Preprocessing:** 
    - Automated cleaning of bidirectional flow data.
    - Label Encoding and Frequency Encoding for high-cardinality features (IPs, Ports).
    - Principal Component Analysis (PCA) integration for SVM optimization.
- **Robust Evaluation:** 
    - Detailed classification reports (Precision, Recall, F1-Score).
    - Visual confusion matrices and feature importance analysis.
    - Class distribution visualization.
- **Logging System:** Comprehensive execution logging for traceability.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

```
botnet-detection/
├── data/
│   ├── CTU-datasets/                    # Input CSV files (not included in the repository)
│   ├── results_EXP1/                    # Output for EXP1
│   ├── results_EXP2_{model}/            # Output for EXP2
│   └── results_EXP3/                    # Output for EXP3
├── botnetDetection_EXP1.py              # Baseline (Single Dataset)
├── botnetDetection_EXP2.py              # Comparative (Cross-Dataset)
└── botnetDetection_EXP3.py              # Multi-Class Aggregation
```

Datasets may be found on [Stratosphere Lab CTU-13 Datasets](https://www.stratosphereips.org/datasets-ctu13) (*CTU University, Czech Republic, 2011*)

To be used based on this they must be renamed according to the following schema:
```
[Index]-[CaptureID]-[BotName]-[Topology].csv
```
obtaining something like `1-42-neris-single.csv`


---

## Experimental Modules

### Experiment 1: Baseline Assessment
**Script:** `botnetDetection_EXP1.py`

This module establishes a performance baseline by processing single dataset files. It splits each dataset internally (80% Train / 20% Test) to verify the separability of the data.

*   **Model:** Random Forest.
*   **Key Functionalities:** 
    *   Generates class distribution plots (`label_count_distribution.png`).
    *   Utilizes a dedicated logging system (`log.txt`) for tracking data cleaning and shape reduction.
    *   Uses stratified splitting to maintain class ratios.

### Experiment 2: Comparative & Cross-Dataset Analysis
**Script:** `botnetDetection_EXP2.py`

The core analytical module designed to test model robustness and generalization. It allows training on one network capture and testing on a completely different one (e.g., Train on *Neris*, Test on *Rbot*).

*   **Models:** Random Forest (default), Decision Tree, MLP, SVM.
*   **Configuration:**
    *   Modify `MODEL_TYPE` in the script to switch classifiers.
    *   For SVM, toggle `ENABLE_PCA_SVM` to enable dimensionality reduction.
    *   For MLP/SVM, the script automatically handles class balancing and scaling.

### Experiment 3: Multi-Class Identification
**Script:** `botnetDetection_EXP3.py`

This module aggregates multiple datasets to perform multi-class classification. Instead of a binary output (Botnet vs. Normal), it distinguishes the specific family of the botnet.

*   **Model:** Random Forest.
*   **Label Mapping:**
    *   `0`: Background / Normal Traffic
    *   `1`: Botnet Type A (e.g., Neris)
    *   `2`: Botnet Type B (e.g., Rbot)
    *   `...`: Botnet Type N

---

## Usage

Run the desired experiment script from the command line:

```bash
# Run Baseline Assessment
python botnetDetection_EXP1.py

# Run Cross-Dataset Analysis (Check configuration inside script first)
python botnetDetection_EXP2.py

# Run Multi-Class Classification
python botnetDetection_EXP3.py
```

## Output & Results

Results are automatically organized into subdirectories based on the experiment and test case.

### Common Output Files
- **report.txt**: Detailed metrics including Accuracy, Precision, Recall, and F1-Score.
- **confMatrix.png**: Visual confusion matrix.
- **log.txt**: (EXP1 & EXP3) Execution logs detailing data shapes and cleaning statistics.

### Model-Specific Plots (RF/DT only)
- **featureImportance.png**: Bar chart ranking the most influential features (Gini importance).
- **dist_top\*.png**: KDE distribution plots comparing Train vs. Test distributions for the top 3 features.

---
**Authors:** Gabriele Aprile, Simone Tassi  
**Course:** University of Bologna – Cybersecurity (2025–2026)