# MIL
Multiple instance learning project for TUM DL seminar WS 23/24

<b>Project assignee:</b> Viktor Prochazka (ProViktor)

<b>Supervisor:</b> Anastasia Litinetskaya (alitinet)

<b>Data:</b> `hlca_subset.h5ad`

This is a subset of the dataset available <a href = 'https://www.nature.com/articles/s41591-023-02327-2'> here </a>. Indices used in the subset are stored??

<b>Meta about split:</b> `dataset_meta.pcl`

<b>How to use:</b>
1. Run all cells of the `exploration_and_split.ipynb` notebook to generate necessary files
2. Get the feel of how nets work on this dataset in  `torch_playground.ipynb` in "playground style"
3. Run hyperparameter search in `optimize_hyper_script.py`
4. Return to `torch_playground.ipynb` to evaluate your results.



<b>Misc:</b>

* Observation: CSR tensors loose precision, use COO whenever you can
* Default torch loaders do not support tensors of differnt lengths, custom loader is implemented, all of the code is adjusted to work with the custom dataloader
![Max aggregation train loss](train_loss.png) 
* Max aggregation is not very useful.