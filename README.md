# 💰 Personal Finance Loan Prediction

A professional, modular PyTorch-based pipeline for predicting personal loan status using tabular financial data. Designed for flexibility, experimentation, and industry-grade extensibility.

---

## 📊 Overview

This project uses a synthetic personal finance dataset to build a **loan approval prediction model**. It processes structured data using deep learning techniques optimized for tabular inputs. The pipeline emphasizes **clarity, reproducibility, modularity**, and **ease of experimentation**.

---

## 🚀 Features

- ✅ Clean and scalable code architecture  
- ✅ Encodes categorical and numerical features using preprocessing pipelines  
- ✅ Configuration-driven training via `config.yaml`  
- ✅ Multi-Layer Perceptron (MLP) with batch normalization and dropout  
- ✅ Logging with timestamped log files and saved encoders/scalers  
- ✅ Modular utility functions for preprocessing, evaluation, and metrics  
- ✅ Simple experiment tracking and reproducibility

---

## 🧪 Project Structure

```bash
PERSONALFINANCECLASSIFIER/
├── checkpoints/             # Saved model weights (.pth)
├── configs/                 # YAML config files
├── data/                    # Input dataset (.csv)
├── logs/                    # Logs, encoders, scalers
├── models/                  # Model definitions (if modularized)
├── PFC/                     # Project core logic (optional submodule)
├── utils/                   # Helper scripts (data, logging, metrics)
│   ├── data_utils.py
│   ├── logger.py
│   ├── metrics.py
│   └── evaluate.py
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Python dependencies
└── README.md                # This file

## Usage
1. Place your CSV in `data/`
2. Adjust configs in `configs/config.yaml`
3. Train:   python train.py
4. Evaluate: python evaluate.py

## Requirements
pip install -r requirements.txt

## 📚 Dataset Acknowledgement
This project uses a **synthetic personal finance dataset** sourced from Kaggle.
**Dataset Source:** [Personal Finance ML Dataset – by miadul on Kaggle](https://www.kaggle.com/datasets/miadul/personal-finance-ml-dataset)  
**Credit:** [@miadul](https://www.kaggle.com/miadul) on Kaggle
> _Note: This dataset contains synthetic financial records for educational and machine learning experimentation purposes._
