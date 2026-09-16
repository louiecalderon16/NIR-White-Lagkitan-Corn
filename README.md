# NIR White Lagkitan Corn

This repository contains the supplementary data, source code, model training notebooks, evaluation records, and reproducibility information for the undergraduate design project:

**Development of a Near-Infrared (NIR) Spectroscopy-Based Classifier for Non-Destructive Sweetness Level of White Lagkitan Corn**

## Repository Structure

### datasets/

Contains the raw and preprocessed spectral datasets used during model development.

- `three_class/raw_dataset.csv` — Raw three-class dataset with 200 samples
- `three_class/preprocessed_dataset.csv` — SNV-preprocessed three-class dataset with sample partition assignments
- `binary/raw_dataset.csv` — Raw binary dataset with 130 samples and 32 scans per sample
- `binary/preprocessed_dataset.csv` — Preprocessed and median-aggregated binary dataset with sample partition assignments

### notebooks/

- `three_class_training.ipynb` — Preprocessing, model tuning, validation, and testing for the three-class development phase
- `binary_training.ipynb` — Preprocessing, model tuning, validation, testing, and final binary model training
- `scan_count_ablation.ipynb` — Scan-count ablation using 1, 5, 10, 15, 20, and 32 scans
- `pca_silhouette_analysis.ipynb` — PCA and silhouette analysis for the three-class and binary datasets

### prototype/

- `datagather_ui.py` — Raspberry Pi data gathering program used during spectral acquisition
- `predict_ui.py` — Deployed Raspberry Pi prototype classification program
- `final_model.joblib` — Final deployed XGBoost binary classification model

### evaluation/

- `functionality_test.xlsx` — Trial-by-trial prototype functionality test records
- `kappa_significance_test.py` — Cohen's Kappa and significance calculation for the 50-sample comparative evaluation

### reproducibility/

- `binary_partition_assignments.csv` — Sample-level binary train, validation, and test assignments
- `three_class_partition_assignments.csv` — Sample-level three-class train, validation, and test assignments
- `environment.md` — Development and Raspberry Pi deployment software environments
- `random_seeds.md` — Random seeds and related reproducibility notes
- `model_info.md` — Final deployed model information and SHA-256 hash

## Notes

- The three-class development phase used the classes **Bland, Average, and Sweet**.
- The final feasibility configuration used **Not Sweet and Sweet**.
- Reference labels for both the three-class and binary configurations were derived from Brix measurements using the defined operational thresholds.
- The final deployed model is preserved in `prototype/final_model.joblib`.
