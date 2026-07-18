# Graph vs Time: METR-LA Traffic Forecasting

An empirical analysis evaluating Spatial-Temporal Graph Neural Networks (STGNN) against standard LSTM baselines for traffic speed forecasting on the Los Angeles Metropolitan (METR-LA) dataset. 

## 📌 Project Overview
This project investigates whether incorporating explicit spatial graph structures (road network distances) improves forecasting accuracy compared to purely temporal sequence models. The evaluation spans multiple prediction horizons (15, 30, and 60 minutes).

## 🚀 Key Features
* **Graph Construction**: Implements a thresholded Gaussian kernel adjacency matrix based on real-world road network distances.
* **STGNN Architecture**: Features diffusion convolution layers coupled with temporal gating, including an 'identity fallback' for ablation studies.
* **Baseline Control**: Includes a strictly temporal LSTM baseline for direct comparison.
* **Masked Evaluation**: Utilizes masked MAE, RMSE, and MAPE metrics to handle missing sensor data accurately.

## 📂 Repository Structure
* \src/data/\ - Graph building (\uild_graph.py\) and chronological data splitting.
* \src/models/\ - Architectures for \stgnn.py\ and \lstm_baseline.py\.
* \scripts/\ - Execution and training pipelines.
* \	ests/\ - Property tests for graph adjacency and model parameter counts.
* \esults/\ - Aggregated \metrics.csv\, raw JSON outputs, and visualization plots (horizon gaps, degree gains).
* \week34_stgnn.ipynb\ - Interactive execution notebook for the complete pipeline.

## 📊 Results & Analysis
The final aggregated metrics and performance visualizations can be found in the \esults/\ directory, highlighting the comparative advantage of STGNNs across extended forecasting horizons.
