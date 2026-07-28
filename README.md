# NIR White Lagkitan Corn Dataset

This repository contains the supplementary data and code for the undergraduate design project 
"Development of a Near-Infrared (NIR) Spectroscopy-Based Classifier for Non-Destructive Sweetness 
Level of White Lagkitan Corn."

## Repository Structure
dataset/
- appendix_preprocessed_dataset1.csv — Preprocessed spectral data, three-class setup (200 samples)
- appendix_preprocessed_dataset2.csv — Preprocessed spectral data, binary setup (130 samples)
- appendix_raw_dataset1.csv — Raw spectral scans, three-class setup (200 scans)
- appendix_raw_dataset2.csv — Raw spectral scans, binary setup (4,160 scans)
source codes/
- predict_ui.py — Deployed prototype system code (also included in full in Appendix Source Code 3.1)
- datagather_ui.py — Data gathering script used during spectral data collection
training notebooks/
- training_pipeline1.ipynb — Model training, preprocessing and evaluation pipeline, three-class setup
- training_pipeline2.ipynb — Model training, preprocessing and evaluation pipeline, binary setup
- 
## Notes

- **Dataset 1** refers to the three-class classification setup (Bland, Average, Sweet).
- **Dataset 2** refers to the binary classification setup (Sweet, Not Sweet).
- Raw datasets contain the unprocessed spectral scans prior to preprocessing.
