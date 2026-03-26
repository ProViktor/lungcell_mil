import numpy as np

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from mil.CellsData import CellsData
from mil.CustomDataloader import CustomLoader

from mil.training_utils import model_run, set_seed, stratified_cv_split
from mil.schemas import RunParams, HyperRunParams
from functools import partial

from mil.models import (
    MIL_model,
    MLP_encoder,
    MeanAggergation,
    MaxAggergation,
    AttentionAggregation,
    GatedAttentionAggregation,
)

from mil import PROJECT_ROOT

from bayes_opt import BayesianOptimization

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Union, Optional
from itertools import chain


class OptimizeHyper:
    def __init__(
        self,
        seeds: Union[Tuple[int], None] = None,
        sparse=True,
        n_epochs=50,
        k_cv=5,
        c_mae_penalize=0,
        note="0",
    ) -> None:
        if seeds is None:
            self.seeds = (0, 37, 42)
        else:
            self.seeds = seeds
        self.sparse = sparse
        self.n_epochs = n_epochs

        if sparse:
            self.input_size = 2000
        else:
            self.input_size = 30

        self.train_set = CellsData(split="train")
        self.validation_set = CellsData(split="val")

        self.k_cv = k_cv
        self.c_mae_penalize = c_mae_penalize
        self.note = note

    def run_search(self, params: HyperRunParams):
        """Re-configures the instance and runs the hyperparameter search."""
        self.sparse = params.sparse
        self.n_epochs = params.n_epochs
        self.k_cv = params.k_cv
        self.c_mae_penalize = params.c_mae_penalize
        self.note = params.note
        self.seeds = params.seeds

        if self.sparse:
            self.input_size = 2000
        else:
            self.input_size = 30

        if params.use_cv:
            f_to_maximize = partial(self.test_model_cv, aggregator=params.aggregator)
        else:
            f_to_maximize = partial(self.test_model, aggregator=params.aggregator)

        self.run_optimizer(
            f_to_maximize=f_to_maximize,
            pbounds=params.pbounds,
            init_points=params.init_points,
            n_iter=params.n_iter,
            random_state=params.random_state,
        )

    def _params_to_settings(self, params: RunParams) -> dict:
        """Converts RunParams to the settings dictionary required for model instantiation."""
        aggregator_classes = {
            "MaxAggergation": MaxAggergation,
            "AttentionAggregation": AttentionAggregation,
            "GatedAttentionAggregation": GatedAttentionAggregation,
            "MeanAggergation": MeanAggergation,
        }
        agg_class = aggregator_classes[params.aggregator]

        agg_settings = {"encoding_size": params.encoding_size}
        if params.aggregator in ["AttentionAggregation", "GatedAttentionAggregation"]:
            agg_settings["attention_hidden_size"] = params.attention_hidden_size

        return {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": params.n_hidden,
                "hidden_size": params.hidden_size,
                "output_size": params.encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": agg_class,
            "aggregator_settings": agg_settings,
        }

    def model_test(
        self,
        run_params: RunParams,
        train_set=None,
        validation_set=None,
        verbose: bool = False,
    ) -> float:
        """Constructs a MIL model from run_params and trains it."""
        if train_set is None:
            train_set = self.train_set
        if validation_set is None:
            validation_set = self.validation_set

        set_seed(run_params.seed)

        settings = self._params_to_settings(run_params)
        encoder_model = settings["encoder"](**settings["encoder_settings"])
        aggregator_model = settings["aggregator"](**settings["aggregator_settings"])

        model = MIL_model(
            instance_encoder=encoder_model, bag_aggregator=aggregator_model
        )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=run_params.lr, weight_decay=run_params.decay
        )

        train_loader = CustomLoader(train_set, batchsize=20)
        validation_loader = CustomLoader(validation_set, batchsize=20)

        ax = None
        if verbose:
            _, ax = plt.subplots()

        _, valid_loss, best_epoch = model_run(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=run_params.num_epochs,
            save_path_prefix=str(
                PROJECT_ROOT / "data" / "max_aggregation_models" / "epoch_"
            ),
            ax=ax,
            plot_title=f"Training of MIL Classifier with {run_params.aggregator}",
            save_weights=verbose,
            sparse=run_params.sparse,
            run_params=run_params.model_dump(),
        )

        return valid_loss[best_epoch]

    def run_optimizer(
        self,
        f_to_maximize,
        pbounds: dict,
        init_points=10,
        n_iter=10,
        verbose=2,
        random_state=0,
    ):
        """Runs the hyperparameter optimization search and logs it.

        Args:
            f_to_maximize (function): Function to maximize, should take model
                hyperparameters as input and return negative loss of the hyperparameter
                 configuration.
            pbounds (dict): Parameter bounds, keys of the dict have to be keyword
                arguments of f_to_maximize.
            init_points (int, optional): Number of random search iteration performed
                by the Bayesian optimizer. Defaults to 10.
            n_iter (int, optional): Number of optimization steps performed by the
                Bayesian optimizer. Defaults to 10.
            verbose (int, optional): Level of verbosity. 0 to turn off. Defaults to 2.
            random_state (int, optional): Random seed. Defaults to 0.
        """

        f_name = (
            f_to_maximize.func.__name__
            if hasattr(f_to_maximize, "func")
            else f_to_maximize.__name__
        )

        print(f"Maximizing {f_name}")

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "runs" / "hyper_optim" / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Log search metadata
        metadata = {
            "f_name": f_name,
            "pbounds": pbounds,
            "init_points": init_points,
            "n_iter": n_iter,
            "random_state": random_state,
            "optimize_hyper_settings": {
                "seeds": self.seeds,
                "sparse": self.sparse,
                "n_epochs": self.n_epochs,
                "k_cv": self.k_cv,
                "c_mae_penalize": self.c_mae_penalize,
                "note": self.note,
            },
        }
        with open(run_dir / "search_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        optimizer = BayesianOptimization(
            f=f_to_maximize,
            pbounds=pbounds,
            verbose=verbose,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=random_state,
        )
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        best_idx = np.argmax([el["target"] for el in optimizer.res])
        best_config = optimizer.res[best_idx]

        best_seed = f_to_maximize(**best_config["params"], seed_search=True)

        # Log best run params
        best_params_dict = best_config["params"].copy()
        if hasattr(f_to_maximize, "keywords"):
            best_params_dict.update(f_to_maximize.keywords)

        best_run_params = self._kwargs_to_run_params(best_params_dict)
        best_run_params.seed = best_seed
        best_run_params.save_json(str(run_dir / "best_run_params.json"))

        p_bounds_str = dict()
        for key, value in pbounds.items():
            p_bounds_str[key] = str(value)
        settings = yaml.dump(p_bounds_str)

        super_settings = dict()
        for attr in ("seeds", "sparse", "n_epochs", "k_cv", "c_mae_penalize", "note"):
            super_settings[attr] = getattr(self, attr)

        super_settings = yaml.dump(super_settings)

        # Convert best_config to native types for YAML dumping
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            return obj

        config_dict = sanitize(best_config)
        result = yaml.dump(config_dict)

        now = datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")

        f_name = (
            f_to_maximize.func.__name__
            if hasattr(f_to_maximize, "func")
            else f_to_maximize.__name__
        )

        doc_string = f"{time}\n\nf: {f_name}\n{super_settings}\n\nn_iter: {n_iter} \ninit_points: {init_points} \n{settings}\nnseed: {best_seed}\nsparse:{self.sparse}\nn_epochs:{self.n_epochs}\n{result}"
        print(doc_string)

        doc_string = doc_string + "\n ---------------------------------\n"
        for el in optimizer.res:
            doc_string = f"{doc_string}\n{sanitize(el)}"

        path_prefix = PROJECT_ROOT / "data" / "hyper_optimization_runs"
        path_prefix.mkdir(parents=True, exist_ok=True)
        now_short = now.strftime("%Y%m%d%H%M%S")
        file_name = f"hyper_optim_run_{now_short}.txt"

        with open(path_prefix / file_name, "w") as f:
            f.write(doc_string)

        # Also save to the run directory
        with open(run_dir / "summary.txt", "w") as f:
            f.write(doc_string)

    def _kwargs_to_run_params(self, kwargs: dict) -> RunParams:
        """Converts float kwargs from Bayesian Optimization to a RunParams instance."""
        params_dict = kwargs.copy()

        # Handle log-scale transformations
        if "log_learning_rate" in params_dict:
            params_dict["lr"] = 10 ** params_dict.pop("log_learning_rate")
        if "log_decay" in params_dict:
            params_dict["decay"] = 10 ** params_dict.pop("log_decay")

        # Round numeric values and update
        int_keys = ["n_hidden", "hidden_size", "encoding_size", "attention_hidden_size"]
        for k in int_keys:
            if k in params_dict and params_dict[k] is not None:
                params_dict[k] = int(np.floor(params_dict[k]))

        # Add class-level defaults if missing
        params_dict.setdefault("sparse", self.sparse)
        params_dict.setdefault("num_epochs", self.n_epochs)
        params_dict.setdefault("seed", 0)  # Placeholder

        # Provide sensible defaults for RunParams required fields if not present
        params_dict.setdefault("n_hidden", 3)
        params_dict.setdefault("hidden_size", 15)
        params_dict.setdefault("encoding_size", 10)
        params_dict.setdefault("lr", 0.001)
        params_dict.setdefault("decay", 0.01)

        if params_dict.get("aggregator") in [
            "AttentionAggregation",
            "GatedAttentionAggregation",
        ]:
            params_dict.setdefault("attention_hidden_size", 10)

        return RunParams(**params_dict)

    def test_model(self, seed_search=False, **kwargs):
        run_params = self._kwargs_to_run_params(kwargs)
        losses_and_seeds = []
        for seed in self.seeds:
            run_params.seed = seed
            loss = self.model_test(run_params)
            losses_and_seeds.append({"loss": loss, "seed": seed})
        losses_and_seeds.sort(key=lambda x: x["loss"])
        return (
            losses_and_seeds[0]["seed"] if seed_search else -losses_and_seeds[0]["loss"]
        )

    def test_model_cv(self, seed_search=False, cv_split=5, c_mae_penalize=2, **kwargs):
        run_params = self._kwargs_to_run_params(kwargs)
        losses = []
        for seed in self.seeds:
            seed_losses = []
            cv_indices = stratified_cv_split(
                dataset=self.train_set, k_cv=cv_split, seed=seed
            )
            for k in range(cv_split):
                validation_set = [self.train_set[idx] for idx in cv_indices[k]]
                train_set_k = [i for i in range(cv_split) if i != k]
                train_set_indices = chain(*[cv_indices[i] for i in train_set_k])
                train_set = [self.train_set[idx] for idx in train_set_indices]

                run_params.seed = seed
                loss = self.model_test(
                    run_params=run_params,
                    train_set=train_set,
                    validation_set=validation_set,
                )
                seed_losses.append(loss)

            # Mean Absolute Error (MAE) around the mean of fold losses
            mae = np.mean(np.abs(seed_losses - np.mean(seed_losses)))
            mean_loss = np.mean(seed_losses) + c_mae_penalize * mae
            losses.append({"loss": mean_loss, "seed": seed})

        losses.sort(key=lambda x: x["loss"])
        return losses[0]["seed"] if seed_search else -losses[0]["loss"]
