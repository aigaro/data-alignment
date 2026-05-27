# Metabolomics Domain Alignment & Transferable ML Pipeline

A Python pipeline for **harmonizing metabolomics datasets across domains** (e.g., batch, instrument, protocol) and training **transferable machine learning models**.

This tool enables robust cross-dataset prediction using multiple **alignment strategies**, a **model zoo**, and built-in **explainability and reporting**.

---

## 🔬 Overview

Metabolomics datasets often suffer from **batch effects and domain shifts**, limiting model transferability. This pipeline:

- Aligns datasets using statistical methods
- Trains models on source data
- Evaluates performance on destination data
- Produces reproducible outputs, plots, and reports

---

## ⚙️ Key Features

### 🧪 Data Harmonization Methods
- **Optimal Transport (OT)** – distribution alignment
- **Bridge mapping**
  - Paired linear regression
  - Nearest-neighbor pseudo-pairing
  - CORAL (covariance alignment)
- **Empirical Bayes (ComBat-style)** – batch correction

### 🤖 Machine Learning
- Models:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
  - Naive Bayes (NB)
  - SGD Classifier
  - KNN
- Hyperparameter tuning via `RandomizedSearchCV`
- Stratified cross-validation
- Metrics:
  - Balanced Accuracy
  - MCC (Matthews Correlation Coefficient)

### 🧠 Advanced Capabilities
- SMOTE (class imbalance handling)
- Feature selection (mutual information)
- SHAP explainability (top features)

### 📊 Outputs
- Aligned datasets (CSV)
- ML results (`ml_results.csv`)
- Plots (publication-quality)
- Markdown summary reports

---

## 📥 Input Data

CSV inputs:

| Argument | Description |
|--------|------------|
| `--sm` | Source matched dataset |
| `--dm` | Destination matched dataset |
| `--snm` | Source unmatched dataset |
| `--dnm` | Destination unmatched dataset |
| `--sml` | Labels for source matched |
| `--dml` | Labels for destination matched |
| `--snml` | Labels for source unmatched (optional) |
| `--dnml` | Labels for destination unmatched (optional) |

Optional:
- `--id_col` → column with sample IDs

---

## 🚀 Quick Start

### Install dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib shap pot


### Run full pipeline

python transform_data.py \
  --sm data/source_matched.csv \
  --sml data/source_labels.csv \
  --dm data/dest_matched.csv \
  --dml data/dest_labels.csv \
  --outdir outputs/run_01 \
  --report --report_plots

### Unmatched-only example

python transform_data.py \
  --snm data/source_unmatched.csv \
  --dnm data/dest_unmatched.csv \
  --snml data/source_labels.csv \
  --dnml data/dest_labels.csv \
  --run_ot --run_bridge --run_eb \
  --outdir outputs/run_02

### Common options

--run_ot                # Run Optimal Transport
--run_bridge            # Run bridge mapping
--run_eb                # Run Empirical Bayes

--use_smote             # Enable SMOTE
--smote_k 5             # Number of neighbors

--keep_k_features 50    # Feature selection (top-k)

--bridge_mode auto|paired|nn|coral

--exclude_models "SVM,NB"


### Output structure

outputs/
├── combined_OT.csv
├── combined_bridge.csv
├── combined_EB.csv
├── combined_baseline.csv
├── combined_normalized.csv
├── ml_results.csv
├── report_summary.md
├── plots/
│   ├── plots_per_method_*.png
│   └── plots_overall_best.png


### Typical workflow

1) Use matched datasets (recommended)
2) Run all alignment methods
3) Compare baseline vs aligned results
4) Evaluate model transferability
5) Inspect SHAP feature importance
6) Select best method/model

### Assumptions & limitations

1) Source and destination must represent the same biological population
2) Differences should be technical (batch effects), not biological
3) Empirical Bayes may remove real signal if poorly applied
4) The pipeline avoids data leakage internally (train-only transformations)

### Reproducibility

1) Fixed random seed: RS=0
2) CLI command saved in cli_command.txt
3) Clean CSV outputs (no unnamed columns)

### License

MIT License © 2026
Sergio Decherchi, Aigar Ottas

### Citation

If used in research, please cite:
"Impact of Matched Samples on Domain Harmonization in Cross-Cohort NMR Metabolomics"

Also cite relevant methods:

Optimal Transport
CORAL
ComBat (Empirical Bayes)

### Authors

Sergio Decherchi
Aigar Ottas

### Pull requests are welcome. For major changes, please open an issue first.
