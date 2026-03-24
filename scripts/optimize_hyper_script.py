from optimize_hyper import OptimizeHyper
"""
Select model type to test, set bounds and run the `run_optimizer` function.
The function stores the its results in directory hyper_optimization_runs

Example:
from optimize_hyper import OptimizeHyper

parameter_search = OptimizeHyper()

f_to_maximize = parameter_search.test_max_model

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

f_to_maximize = parameter_search.test_max_model

pbounds = {
    "hidden_size": (2, 50.5),
    "n_hidden": (1, 3.5),
    "log_learning_rate": (-4, -1),
    "log_decay": (-3, 1),
}

parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20)

parameter_search = OptimizeHyper(sparse=False, n_epochs=80)

f_to_maximize = parameter_search.test_attention_model

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

f_to_maximize = parameter_search.test_gated_attention_model

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


parameter_search = OptimizeHyper(sparse=True, n_epochs=30, c_std_penalize=0.1, note='3 layer post - mean agg')

f_to_maximize = parameter_search.test_attention_model_cv

pbounds = {
    "hidden_size": (35, 45), 
    "n_hidden": (1, 3.99),
    "encoding_size": (20, 50.5),
    "log_learning_rate": (-2, -1.99),
    "log_decay": (-2.5, -1),
}
parameter_search.run_optimizer(f_to_maximize, pbounds, init_points=20, n_iter=20, random_state=1)