# 🚦 Graph vs Time: METR-LA Traffic Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An empirical analysis and modeling pipeline evaluating **Spatial-Temporal Graph Neural Networks (STGNN)** against standard LSTM baselines for traffic speed forecasting on the Los Angeles Metropolitan (METR-LA) dataset.

---

## 📌 The Core Question
**Does incorporating explicit spatial graph structures (road network distances) fundamentally improve forecasting accuracy over purely temporal sequence models?** 

This project tackles that question across multiple prediction horizons (15, 30, and 60 minutes) to determine where STGNNs excel and where traditional models plateau.

## 🚀 Key Architectural Features

*   🗺️ **Graph Construction:** Engineered a thresholded Gaussian kernel adjacency matrix mapping real-world road network distances.
*   🧠 **STGNN Pipeline:** Implemented diffusion convolution layers coupled with temporal gating (includes an identity fallback for ablation).
*   ⏱️ **Baseline Control:** Developed a strictly temporal LSTM baseline for direct, controlled comparison.
*   🛡️ **Robust Evaluation:** Built masked MAE, RMSE, and MAPE metrics to accurately handle and bypass missing sensor data.

---

## 📊 Performance Results

*Evaluation metrics aggregated across masked sensors.*

| Model | Horizon | MAE | RMSE | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM (Baseline)** | 15 min | *value* | *value* | *value* |
| **STGNN (Identity)**| 15 min | *value* | *value* | *value* |
| **STGNN (Road)** | 15 min | *value* | *value* | *value* |

> **Note:** Comprehensive performance visualizations, including horizon gaps and degree gains, are available in the \/results\ directory.

---

## 💻 Quick Start & Execution

Explore the full pipeline interactively via the provided Jupyter Notebook:

1. Clone the repository.
2. Install dependencies (requires PyTorch and standard data science libraries).
3. Run the execution notebook:
   \\\ash
   jupyter notebook week34_stgnn.ipynb
   \\\

## 📂 Repository Architecture

*   **\src/data/\** — Adjacency building (\uild_graph.py\) and chronological splits.
*   **\src/models/\** — Network architectures (\stgnn.py\, \lstm_baseline.py\).
*   **\scripts/\** — Execution and training pipelines.
*   **\	ests/\** — Property tests for graph adjacency and model parameter parity.
*   **\esults/\** — Aggregated metrics (\metrics.csv\), raw JSON, and visual plots.
