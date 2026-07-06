# Multi-Agent Trajectory Extraction & Analysis Pipeline 🚗📊

## Overview
This repository contains a modular data science framework designed to extract, process, and analyze multi-agent motion and autonomous driving datasets. 

Instead of hardcoding for a specific source, this project implements a unified pipeline capable of handling diverse international datasets. The system ingests raw vehicle/pedestrian trajectories, aligns varying temporal frequencies, and couples them with HD semantic maps to analyze complex traffic behaviors and critical maneuvers.

## Project Structure
The pipeline is designed with modularity in mind to support seamless integration of new data sources:
```text
├── data/                  # Local data storage (Git-ignored)
│   ├── raw/               # Raw datasets (INTERACTION, Waymo, etc.)
│   └── processed/         # Standardized unified format
├── src/                   # Source code
│   ├── extractors/        # Dataset-specific parsing modules
│   ├── transformation/    # Normalization and feature engineering
│   └── visualization/     # Map plotting and trajectory rendering
├── notebooks/             # Exploratory Data Analysis (EDA)
└── README.md
