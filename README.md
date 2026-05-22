# Credit Card Fraud Detection
> Unsupervised anomaly detection using a Deep Autoencoder + Isolation Forest ensemble with SHAP explainability.

---

## Results

| Metric | Value |
|--------|-------|
| AUROC | **0.955** |
| F1 Score | **0.637** |
| False Positive Rate | **0.45%** |
| Reconstruction Error Gap (fraud vs normal) | **65×** |

> Fully unsupervised — no fraud labels used during training.

---

## How it works

The core idea: train an autoencoder **only on normal transactions**. It learns to reconstruct normal behavior well. When it sees a fraudulent transaction, reconstruction error spikes — that spike is the anomaly signal.

```
Raw Transactions
      │
      ▼
Feature Engineering (log-amount, time-delta, z-score)
      │
      ▼
┌─────────────────────────────────────┐
│           Ensemble Scoring          │
│                                     │
│  Autoencoder  ──► Reconstruction   │
│  (PyTorch)         Error Score      │
│                         │           │
│  Isolation   ──► Isolation          │
│  Forest           Score             │
│                         │           │
│         Weighted Combine            │
│         (0.75 AE + 0.25 IF)         │
│         p1–p99 Robust Normalize     │
└─────────────────────────────────────┘
      │
      ▼
Threshold Tuning (max F1 @ FPR ≤ 1%)
      │
      ▼
Binary Prediction + SHAP Explainability
```

---

## Project Structure

```
credit_fraud_detection/
├── data/
│   ├── raw/                  # creditcard.csv (gitignored)
│   └── processed/            # parquet files (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py         # load, split dataset
│   │   ├── preprocessor.py   # scaling, normalization
│   │   └── feature_engineer.py  # log-amount, time-delta, z-score
│   ├── models/
│   │   ├── autoencoder.py    # PyTorch encoder-decoder
│   │   ├── isolation_forest.py  # sklearn wrapper
│   │   └── ensemble.py       # weighted score fusion
│   ├── training/
│   │   ├── train_autoencoder.py  # train loop, early stopping
│   │   └── train_iforest.py
│   ├── evaluation/
│   │   ├── metrics.py        # AUPRC, F1, recall@precision, FPR
│   │   ├── threshold.py      # PR curve, optimal threshold, plots
│   │   └── explainability.py # SHAP for AE and IF
│   └── utils/
│       ├── config.py
│       └── logger.py
├── saved_models/             # gitignored
├── outputs/
│   ├── plots/                # PR curve, SHAP, score distribution, loss
│   └── reports/              # metrics.json
├── main.py                   # end-to-end pipeline runner
├── config.yaml               # all hyperparameters
└── requirements.txt
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/credit-fraud-detection.git
cd credit-fraud-detection
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:
```
data/raw/creditcard.csv
```

**4. Run**
```bash
python main.py
```

That's it. One command runs the full pipeline — preprocessing → training → scoring → evaluation → SHAP plots.

---

## Configuration

All hyperparameters live in `config.yaml`. No hardcoded values anywhere.

```yaml
autoencoder:
  hidden_dims: [128, 64, 32]   # encoder path — decoder mirrors automatically
  dropout: 0.1
  learning_rate: 0.0005
  batch_size: 512
  epochs: 150
  early_stopping_patience: 15

isolation_forest:
  n_estimators: 200
  contamination: 0.0017        # actual fraud rate in dataset

ensemble:
  ae_weight: 0.75
  if_weight: 0.25

evaluation:
  target_recall: 0.75
```

---

## Outputs

After running, `outputs/` contains:

| File | Description |
|------|-------------|
| `plots/training_loss.png` | Autoencoder train vs val loss |
| `plots/precision_recall_curve.png` | PR curve with chosen threshold marked |
| `plots/score_distribution.png` | Fraud vs normal score separation |
| `plots/shap_summary_autoencoder.png` | SHAP beeswarm — feature impact |
| `plots/shap_summary_iforest.png` | SHAP beeswarm — Isolation Forest |
| `plots/shap_importance_*.png` | Global feature importance bar charts |
| `plots/shap_single_fraud_N.png` | Per-transaction feature contribution |
| `reports/metrics.json` | All evaluation metrics |

---

## Why these metrics?

**AUROC 0.955** — the model correctly ranks 95.5% of fraudulent transactions above normal ones. Measures ranking quality globally.

**AUPRC** — primary metric for imbalanced problems. Accuracy is meaningless here (predicting "no fraud" always gives 99.8% accuracy). AUPRC measures precision-recall tradeoff across all thresholds.

**FPR 0.45%** — only 256 out of 56,863 normal transactions were incorrectly flagged. This is the customer experience cost.

**Why not accuracy?** With 0.17% fraud rate, a model predicting "never fraud" scores 99.83% accuracy and catches zero fraud. Accuracy is the wrong metric here.

---

## Key design decisions

**Train autoencoder on normal only** — the model never sees fraud during training. It learns the manifold of normal behavior. Fraud deviates from this manifold, producing high reconstruction error.

**p1–p99 robust normalization** — before combining AE and IF scores, each is clipped at the 1st and 99th percentile then scaled to [0,1]. This prevents a handful of extreme outliers from collapsing the threshold.

**Max-F1 threshold with FPR constraint** — instead of a fixed 0.5 threshold, the optimal threshold is found by maximizing F1 subject to FPR ≤ 1%. This reflects real business constraints.

**SHAP explainability** — every anomaly flag comes with a feature-level explanation. The SHAP waterfall plot for individual transactions shows exactly which features drove the anomaly score up or down.

---

## Tech Stack

- **Python 3.12**
- **PyTorch** — autoencoder architecture and training
- **scikit-learn** — Isolation Forest, metrics, preprocessing
- **SHAP** — model explainability
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — visualization

---

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases (0.17%). Features V1–V28 are PCA-transformed for confidentiality. Only `Amount` and `Time` are in original form.

---

## Author
**Abishek**
