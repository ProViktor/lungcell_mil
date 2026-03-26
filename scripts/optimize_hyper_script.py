import os

os.environ["WANDB_SILENT"] = "true"
from optimize_hyper import OptimizeHyper
from functools import partial

"""
Select model type to test, set bounds and run the `run_optimizer` function.
The function stores its results in directory hyper_optimization_runs and logs best parameters to runs/hyper_optim/.

Example:
from optimize_hyper import OptimizeHyper
from functools import partial

parameter_search = OptimizeHyper()

f_to_maximize = partial(parameter_search.test_model, aggregator="MeanAggergation")

pbounds = {
    "hidden_size": (2, 50.5),
    "n_hidden": (1, 3.5),
    "log_learning_rate": (-4, -1),
    "log_decay": (-3, 1),
}

parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20)
"""

"""
parameter_search = OptimizeHyper(sparse=False, n_epochs=80)

f_to_maximize = partial(parameter_search.test_model, aggregator="MaxAggergation")

pbounds = {
    "hidden_size": (2, 50.5),
    "n_hidden": (1, 3.5),
    "log_learning_rate": (-4, -1),
    "log_decay": (-3, 1),
}

parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20)

parameter_search = OptimizeHyper(sparse=False, n_epochs=80)

f_to_maximize = partial(parameter_search.test_model, aggregator="AttentionAggregation")

pbounds = {
    "hidden_size": (2, 50.5),
    "n_hidden": (1, 3.5),
    "encoding_size": (2, 50.5),
    "attention_hidden_size": (2, 50.5),
    "log_learning_rate": (-4, -1),
    "log_decay": (-3, 1),
}

parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20)

parameter_search = OptimizeHyper(sparse=False, n_epochs=80)

f_to_maximize = partial(parameter_search.test_model, aggregator="GatedAttentionAggregation")

pbounds = {
    "hidden_size": (2, 50.5),
    "n_hidden": (1, 3.5),
    "encoding_size": (2, 50.5),
    "attention_hidden_size": (2, 50.5),
    "log_learning_rate": (-4, -1),
    "log_decay": (-3, 1),
}

parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20)
"""

parameter_search = OptimizeHyper(sparse=False, n_epochs=30, c_mae_penalize=0.1, note="")

f_to_maximize = partial(
    parameter_search.test_model_cv, aggregator="AttentionAggregation"
)

pbounds = {
    "hidden_size": (35, 45),
    "n_hidden": (1, 3.99),
    "encoding_size": (20, 50.99),
    "log_learning_rate": (-2, -1.99),
    "log_decay": (-2, -1),
}
parameter_search.run_optimizer(
    f_to_maximize, pbounds, init_points=5, n_iter=5, random_state=42
)
