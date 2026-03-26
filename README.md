# MIL
Multiple instance learning project for TUM DL seminar WS 23/24
This project tries to predict whether a patient did or did not have covid based on the gene expressions in their lung cells.

<b>Project assignee:</b> Viktor Prochazka (ProViktor)

<b>Supervisor:</b> Anastasia Litinetskaya (alitinet)

<b>Data:</b> `hlca_subset.h5ad`

This is a subset of the dataset available <a href = 'https://www.nature.com/articles/s41591-023-02327-2'> here </a>.

<b>Meta about split:</b> `dataset_meta.pcl`

<b>How to use:</b>
0. Installation with `uv sync`
1. Run all cells of the `exploration_and_split.ipynb` notebook to generate necessary files
2. Get the feeling for how nets work on this dataset in  `torch_playground.ipynb` in "playground style"
3. Run hyperparameter search:
    - Edit and run `uv run python scripts/optimize_hyper_script.py`
4. Return to `torch_playground.ipynb` to evaluate your results.




### Logging
Search metadata and the best found parameters are automatically saved to `runs/hyper_optim/run_TIMESTAMP/`.