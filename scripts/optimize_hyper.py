import numpy as np

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from mil.CellsData import CellsData
from mil.CustomDataloader import CustomLoader

from mil.training_utils import model_run, set_seed, stratified_cv_split
from mil.schemas import RunParams

from mil.models import (
    MIL_model,
    MLP_encoder,
    MaxAggergation,
    AttentionAggregation,
    GatedAttentionAggregation,
)

from mil import PROJECT_ROOT

from bayes_opt import BayesianOptimization

import yaml
import ast
from datetime import datetime
from typing import Tuple, Union
from itertools import chain


class OptimizeHyper:
    def __init__(
        self,
        seeds: Union[Tuple[int], None] = None,
        sparse=True,
        n_epochs=50,
        k_cv=5,
        c_std_penalize=0,
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
        self.c_std_penalize = c_std_penalize
        self.note = note

    def model_test(
        self,
        encoder: nn.Module,
        encoder_settings: dict,
        aggregator: nn.Module,
        aggregator_settings: dict,
        learning_rate: float = 0.001,
        decay: float = 0.01,
        seed: int = 37,
        verbose: bool = False,
        sparse: bool = True,
        train_set=None,
        validation_set=None,
    ):
        """Constructs a MIL model from encoder and aggregator and trains it.
        Returns the lowest loss on the validation dataset

        Args:
            encoder (nn.Module): Bag instance encoder class
            encoder_settings (dict): Hyperparameter settings to instantiate the encoder model.
            aggregator (nn.module): Aggregator model class
            aggregator_settings (dict): Hyperparameter settings to instantiate the aggregator model.
            learning_rate (float, optional): Learning rate for the optimizer to use. Defaults to 0.001.
            decay (float, optional): Weight decay (reguralization) for the optimizer to use. Defaults to 0.01.
            seed (int, optional): Random seed for model initialization. Defaults to 37.
            verbose (bool, optional): Verbosity. Defaults to False.
            sparse (bool, optional): If the model should use the sparse data in the HLCA dataset. Defaults to True.

        Returns:
            float: minimal loss on the validation dataset
        """

        if train_set is None:
            train_set = self.train_set
        if validation_set is None:
            validation_set = self.validation_set

        set_seed(seed)

        encoder_model = encoder(**encoder_settings)
        aggregator_model = aggregator(**aggregator_settings)

        model = MIL_model(
            instance_encoder=encoder_model, bag_aggregator=aggregator_model
        )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

        num_epochs = self.n_epochs

        train_loader = CustomLoader(train_set, batchsize=20)
        validation_loader = CustomLoader(validation_set, batchsize=20)
        if verbose:
            fig, ax = plt.subplots()

        if not verbose:
            ax = None

        run_params = RunParams(
            aggregator=aggregator.__name__,
            n_hidden=encoder_settings["n_hidden"],
            hidden_size=encoder_settings["hidden_size"],
            encoding_size=encoder_settings["output_size"],
            seed=seed,
            lr=learning_rate,
            decay=decay,
            attention_hidden_size=aggregator_settings.get("attention_hidden_size"),
            sparse=sparse,
            num_epochs=num_epochs,
        )

        train_loss, valid_loss, best_epoch = model_run(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            save_path_prefix=str(
                PROJECT_ROOT / "data" / "max_aggregation_models" / "epoch_"
            ),
            ax=ax,
            plot_title="Training of MIL Classifier with Max Aggregation",
            save_weights=verbose,
            sparse=sparse,
            run_params=run_params.model_dump(),
        )

        # classic loss
        min_loss = valid_loss[best_epoch]

        # smoothened loss
        n = len(valid_loss)
        """
        smooth = []
        smooth_k = 3
        for i in range(n-smooth_k):
            sub_range = valid_loss[i: i+smooth_k]
            smooth.append(np.mean(sub_range))
        
        min_loss = np.min(smooth)
        """

        return min_loss

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

        print(f"Maximizing {f_to_maximize.__name__}")

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

        p_bounds_str = dict()
        for key, value in pbounds.items():
            p_bounds_str[key] = str(value)
        settings = yaml.dump(p_bounds_str)

        super_settings = dict()
        for attr in ("seeds", "sparse", "n_epochs", "k_cv", "c_std_penalize", "note"):
            super_settings[attr] = getattr(self, attr)

        super_settings = yaml.dump(super_settings)

        config_dict = ast.literal_eval(str(best_config))
        result = yaml.dump(config_dict)

        now = datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")

        f_name = f_to_maximize.__name__

        doc_string = f"{time}\n\nf: {f_name}\n{super_settings}\n\nn_iter: {n_iter} \ninit_points: {init_points} \n{settings}\nnseed: {best_seed}\nsparse:{self.sparse}\nn_epochs:{self.n_epochs}\n{result}"
        print(doc_string)

        doc_string = doc_string + "\n ---------------------------------\n"
        for el in optimizer.res:
            doc_string = f"{doc_string}\n{el}"

        path_prefix = PROJECT_ROOT / "data" / "hyper_optimization_runs"
        path_prefix.mkdir(parents=True, exist_ok=True)
        now_short = now.strftime("%Y%m%d%H%M%S")
        file_name = f"hyper_optim_run_{now_short}.txt"

        with open(path_prefix / file_name, "w") as f:
            f.write(doc_string)

    def test_model(self, settings, seed_search=False):
        losses_and_seeds = []
        for seed in self.seeds:
            settings["seed"] = seed
            loss = self.model_test(**settings)
            losses_and_seeds.append({"loss": loss, "seed": seed})
        losses_and_seeds.sort(key=lambda x: x["loss"])
        if seed_search:
            otp = losses_and_seeds[0]["seed"]
        else:
            otp = -losses_and_seeds[0]["loss"]

        return otp

    def test_model_cv(self, settings, seed_search=False, cv_split=5, c_std_penalize=2):
        losses = []
        for seed in self.seeds:
            seed_losses = []
            cv_indices = stratified_cv_split(
                dataset=self.train_set, k_cv=cv_split, seed=seed
            )
            for k in range(cv_split):
                validation_set = []
                for idx in cv_indices[k]:
                    validation_set.append(self.train_set[idx])

                train_set_k = [i for i in range(cv_split) if i != k]
                train_set_indices = chain(*[cv_indices[i] for i in train_set_k])
                train_set = []
                for idx in train_set_indices:
                    train_set.append(self.train_set[idx])

                loss = self.model_test(
                    **settings, train_set=train_set, validation_set=validation_set
                )
                seed_losses.append(loss)

            mean_loss = np.mean(seed_losses) + c_std_penalize * np.std(seed_losses)
            losses.append({"loss": mean_loss, "seed": seed})

        losses.sort(key=lambda x: x["loss"])

        if seed_search:
            otp = losses[0]["seed"]
        else:
            otp = -losses[0]["loss"]

        return otp

    def test_max_model(
        self,
        n_hidden=3,
        hidden_size=15,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """
        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))

        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": 1,
                "input_size": self.input_size,
            },
            "aggregator": MaxAggergation,
            "aggregator_settings": {"use_sigmoid": True},
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model(settings, seed_search)
        return otp

    def test_max_model_cv(
        self,
        n_hidden=3,
        hidden_size=15,
        encoding_size=20,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """
        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))
        encoding_size = int(np.floor(encoding_size))
        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": MaxAggergation,
            "aggregator_settings": {
                "post_process": True,
                "encoding_size": encoding_size,
            },
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model_cv(settings=settings, seed_search=seed_search, cv_split=5)
        return otp

    def test_attention_model(
        self,
        n_hidden=3,
        hidden_size=15,
        encoding_size=10,
        attention_hidden_size=10,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """

        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))
        encoding_size = int(np.floor(encoding_size))
        attention_hidden_size = int(np.floor(attention_hidden_size))

        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": AttentionAggregation,
            "aggregator_settings": {
                "encoding_size": encoding_size,
                "attention_hidden_size": attention_hidden_size,
            },
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model_cv(settings=settings, seed_search=seed_search, cv_split=5)
        return otp

    def test_attention_model_cv(
        self,
        n_hidden=3,
        hidden_size=15,
        encoding_size=10,
        attention_hidden_size=10,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """

        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))
        encoding_size = int(np.floor(encoding_size))
        attention_hidden_size = int(np.floor(attention_hidden_size))

        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": AttentionAggregation,
            "aggregator_settings": {
                "encoding_size": encoding_size,
                "attention_hidden_size": attention_hidden_size,
            },
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model_cv(settings, seed_search)
        return otp

    def test_gated_attention_model(
        self,
        n_hidden=3,
        hidden_size=15,
        encoding_size=10,
        attention_hidden_size=10,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """

        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))
        encoding_size = int(np.floor(encoding_size))
        attention_hidden_size = int(np.floor(attention_hidden_size))

        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": GatedAttentionAggregation,
            "aggregator_settings": {
                "encoding_size": encoding_size,
                "attention_hidden_size": attention_hidden_size,
            },
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model(settings, seed_search)
        return otp

    def test_gated_attention_model_cv(
        self,
        n_hidden=3,
        hidden_size=15,
        encoding_size=10,
        attention_hidden_size=10,
        log_learning_rate=-3.0,
        log_decay=-2.0,
        seed_search=False,
    ):
        """Returns -minimal_loss (maximization is equivalent to loss minimization)

        Args:
            n_hidden (int, optional): _description_. Defaults to 3.
            hidden_size (int, optional): _description_. Defaults to 15.
            lr (float, optional): _description_. Defaults to 0.001.
            decay (float, optional): _description_. Defaults to 0.01.
        """

        n_hidden = int(np.floor(n_hidden))
        hidden_size = int(np.floor(hidden_size))
        encoding_size = int(np.floor(encoding_size))
        attention_hidden_size = int(np.floor(attention_hidden_size))

        settings = {
            "encoder": MLP_encoder,
            "encoder_settings": {
                "n_hidden": n_hidden,
                "hidden_size": hidden_size,
                "output_size": encoding_size,
                "input_size": self.input_size,
            },
            "aggregator": GatedAttentionAggregation,
            "aggregator_settings": {
                "encoding_size": encoding_size,
                "attention_hidden_size": attention_hidden_size,
            },
            "learning_rate": 10**log_learning_rate,
            "decay": 10**log_decay,
            "sparse": self.sparse,
        }

        otp = self.test_model_cv(settings, seed_search)
        return otp
