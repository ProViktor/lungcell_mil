# Hyperparameter Optimization Scripts

This directory contains tools for optimizing the Multiple Instance Learning (MIL) models using Bayesian Optimization.

## Core Components

- `optimize_hyper.py`: Contains the `OptimizeHyper` class which manages the data, model instantiation, and the Bayesian Optimization loop.
- `optimize_hyper_script.py`: The entry point script where you define search bounds and select the model/evaluation strategy.

## Evaluation Strategies

The search can be performed using two primary methods within the `OptimizeHyper` class:

### 1. `test_model` (Fixed Validation Split)
- **Mechanism**: Trains the model on the `train_set` and evaluates it on the fixed `validation_set`.
- **Pros**: Significantly faster than cross-validation.
- **Cons**: High risk of **validation set overfitting**. Because the Bayesian Optimizer evaluates dozens of configurations against the exact same validation split, it may select hyperparameters that perform well on that specific subset by chance, rather than finding a truly generalizable model.

### 2. `test_model_cv` (K-Fold Cross-Validation)
- **Mechanism**: Performs Stratified K-Fold cross-validation on the `train_set`. It splits the training data into $K$ folds, training on $K-1$ and validating on the remaining fold, repeating this $K$ times.
- **Stability Penalization**: This method includes a stability mechanism controlled by the `c_mae_penalize` parameter. The final loss returned is calculated using the Mean Absolute Deviation (MAD) of the fold losses:
  $$\text{Score} = \text{mean}(\text{losses}) + c_{\text{mae\_penalize}} \times \text{mean}(|\text{losses} - \text{mean}(\text{losses})|)$$
  This penalizes hyperparameter configurations that are inconsistent across folds. Using the absolute deviation is more robust than the standard deviation for small sample sizes (like 5 folds). A high variance in performance suggests the configuration might be sensitive to specific data splits, and adding this penalty encourages the optimizer to find "stable" configurations that perform consistently well across all folds.
- **Pros**: Much more robust. It provides a better estimate of how the model generalizes to unseen data and significantly reduces the risk of overfitting to a single split.
- **Cons**: $K$ times slower than `test_model` per iteration.

### 3. `optimize_hyper_hydra.py` (Hydra Entry Point)
- **Mechanism**: Uses Hydra to manage configurations and parallel execution.
- **Parallel Runs**: Supports running multiple searches in parallel using the `-m` (multirun) flag and the Joblib launcher.
- **Config Management**: Search bounds and global settings are stored in `configs/`.

## Action Flow

1. **Initialization**: `OptimizeHyper` loads the `CellsData` and determines the `input_size` based on whether `sparse` mode is enabled.
2. **Setup**:
    - **Script Mode**: In `optimize_hyper_script.py`, a `partial` function is created to bind a specific `aggregator`.
    - **Hydra Mode**: In `optimize_hyper_hydra.py`, Hydra loads the specified YAML configuration and validates it against the `HyperRunParams` schema.
3. **Optimization Loop**: The `run_search` (or `run_optimizer`) method starts the `BayesianOptimization` process:
...
4. **Finalization**: ...

## How to use Hydra Parallel Optimization

Ensure you have the dependencies installed:
```bash
uv add hydra-core hydra-joblib-launcher
```

### Running a single search:
```bash
python scripts/optimize_hyper_hydra.py aggregator=attention
```

### Running multiple searches in parallel:
```bash
# This will launch both Attention and Max aggregation searches in parallel (n_jobs=2 by default)
python scripts/optimize_hyper_hydra.py -m aggregator=attention,max
```

### Overriding parameters from CLI:
```bash
python scripts/optimize_hyper_hydra.py common_settings.n_epochs=50 aggregator=gated_attention
```
