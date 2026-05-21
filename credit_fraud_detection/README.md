# Credit Card Fraud Detection

Anomaly detection system using an Autoencoder + Isolation Forest ensemble with SHAP explainability.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it at `data/raw/creditcard.csv`.

## Run

```bash
python main.py
```

## Structure

| Folder | Purpose |
|--------|---------|
| `src/data/` | Loading, preprocessing, feature engineering |
| `src/models/` | Autoencoder, Isolation Forest, Ensemble |
| `src/training/` | Training loops |
| `src/evaluation/` | Metrics, threshold tuning, SHAP |
| `notebooks/` | EDA and experiments |
| `outputs/` | Plots and reports |

## Metrics

Primary metric: **AUPRC** (Area Under Precision-Recall Curve).
Also reports F1, Recall @ 95% Precision, and False Positive Rate.
