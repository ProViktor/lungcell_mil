# MIL
Multiple instance learning project for TUM DL seminar WS 23/24
This project tries to predict whether a patient did or did not have covid based on the gene expressions in their lung cells.

<b>Project assignee:</b> Viktor Prochazka (ProViktor)

<b>Supervisor:</b> Anastasia Litinetskaya (alitinet)

<b>Data:</b> `hlca_subset.h5ad`

This is a subset of the dataset available <a href = 'https://www.nature.com/articles/s41591-023-02327-2'> here </a>.

<b>Meta about split:</b> `dataset_meta.pcl`

<b>How to use:</b>
1. Run all cells of the `exploration_and_split.ipynb` notebook to generate necessary files
2. Get the feel of how nets work on this dataset in  `torch_playground.ipynb` in "playground style"
3. Run hyperparameter search:
    - Simple: Edit and run `scripts/optimize_hyper_script.py`
    - Advanced/Parallel: Use Hydra with `scripts/optimize_hyper_hydra.py` (see below)
4. Return to `torch_playground.ipynb` to evaluate your results.

## Hyperparameter Optimization with Hydra

The project supports parallel hyperparameter optimization using Hydra. Configuration is managed in the `configs/` directory.

### Installation
```bash
uv add hydra-core hydra-joblib-launcher
```

### Running Optimization
To run a single optimization for a specific aggregator:
```bash
python scripts/optimize_hyper_hydra.py aggregator=attention
```

To launch multiple optimization runs in parallel (e.g., using both Attention and Max aggregation):
```bash
python scripts/optimize_hyper_hydra.py -m aggregator=attention,max
```

Parameters can be overridden via command line:
```bash
python scripts/optimize_hyper_hydra.py common_settings.n_epochs=50 aggregator=gated_attention
```

### Logging
Search metadata and the best found parameters are automatically saved to `runs/hyper_optim/run_TIMESTAMP/`.

<b>Misc:</b>

* Observation: CSR tensors loose precision, use COO whenever you can
* Default torch loaders do not support tensors of differnt lengths, custom loader is implemented, all of the code is adjusted to work with the custom dataloader
* Max aggregation is not very useful.
![Max aggregation train loss](train_loss.png) 