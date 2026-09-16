# Random Seeds

## Three-Class Development

- Train/validation/test split: `random_state=42`
- Optuna TPE sampler: `seed=42`
- Repeated stratified cross-validation: `random_state=42`
- Cross-validation configuration: 5 folds × 3 repeats

## Binary Development

- Train/validation/test split: `random_state=42`
- Optuna TPE sampler: `seed=42`
- Repeated stratified cross-validation: `random_state=42`
- Cross-validation configuration: 10 folds × 3 repeats

## Model Random State Note

During hyperparameter tuning, applicable model definitions used `random_state=42`.

However, the final XGBoost models used for validation/test reconstruction and final full-data export were created using `study_xgb.best_params`. Since fixed parameters such as `random_state=42` were not included in Optuna's `best_params`, the final exported model did not explicitly receive `random_state=42`.

The exact deployed model artifact is preserved in `prototype/final_model.joblib` and identified by its SHA-256 hash in `model_info.md`.
