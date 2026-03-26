import os
from optimize_hyper import OptimizeHyper
from mil.schemas import HyperRunParams

os.environ["WANDB_SILENT"] = "true"


def main():
    # Define hyperparameter search parameters using the HyperRunParams schema
    params = HyperRunParams(
        aggregator="AttentionAggregation",
        use_cv=True,
        sparse=False,
        n_epochs=30,
        c_mae_penalize=0.1,
        note="Manual optimization run",
        init_points=5,
        n_iter=5,
        random_state=42,
        pbounds={
            "hidden_size": (35, 45),
            "n_hidden": (1, 3.99),
            "encoding_size": (20, 50.99),
            "log_learning_rate": (-2, -1.99),
            "log_decay": (-2, -1),
            "attention_hidden_size": (5, 20),
        },
    )

    # Initialize and run search
    parameter_search = OptimizeHyper()
    parameter_search.run_search(params)


if __name__ == "__main__":
    main()
