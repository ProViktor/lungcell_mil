from mil.models import MIL_model, MLP_encoder, MeanAggergation, MaxAggergation, AttentionAggregation, GatedAttentionAggregation
from mil.training_utils import model_run, set_seed
from mil.evaluation_utils import evaluate_accuracy
from mil.CustomDataloader import CustomLoader
from mil.CellsData import CellsData

from mil import PROJECT_ROOT

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# Set to True if you want to use original sparse matrix data, False if you want low dimensional embeddings
use_sparse_data = True


if use_sparse_data:
    INPUT_SIZE = 2000
    BAG_KEY = 'bag'
    NUM_EPOCHS = 30
else:
    INPUT_SIZE = 30
    BAG_KEY = 'bag_embed'
    NUM_EPOCHS = 30



train_set = CellsData(split="train")
val_set = CellsData(split="val")
test_set = CellsData(split="test")

n_bags = sum([len(el) for el in (train_set, val_set, test_set)])
print(f"The entire datatset contains {n_bags} bags.")

train_loader = CustomLoader(train_set, batchsize=20)
validation_loader = CustomLoader(val_set, batchsize=20)
test_loader = CustomLoader(test_set, batchsize=20)




def run_evaluation(encoder: nn.Module, aggregator: nn.Module, lr: float, decay: float, plot_title: str, path_prefix: str):
    model = MIL_model(instance_encoder=encoder, bag_aggregator=aggregator)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    fig, ax = plt.subplots()


    train_loss, valid_loss, best_epoch = model_run(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        save_path_prefix=path_prefix,
        ax=ax,
        plot_title=plot_title,
        sparse=use_sparse_data,
    )

    path = path_prefix + f"{best_epoch}.torch"
    model = torch.load(path, weights_only=False)
    evaluate_accuracy(model=model, dataset=test_set, bag_key = BAG_KEY, mode = 'Test')


run_params = {}
if use_sparse_data:
    run_params = {
        "n_hidden": 3,
        "hidden_size": 10,
        "lr": 10**-2,
        "decay": 10**-2,
        "encoding_size": 10,
        "seed": 37,
    }
else:
    run_params = {
    "n_hidden": 1,
    "hidden_size": 30,
    "lr": 10**-2,
    "decay": 10**-1.7005985187830885,
    "seed": 27,
    "encoding_size": 30,
    }

set_seed(run_params["seed"])
encoder = MLP_encoder(
    n_hidden=run_params["n_hidden"],
    hidden_size=run_params["hidden_size"],
    output_size=run_params["encoding_size"],
    input_size=INPUT_SIZE
)
aggregator = MeanAggergation(post_process=True, encoding_size=run_params["encoding_size"])

run_evaluation(
    encoder,
    aggregator,
    run_params["lr"],
    run_params["decay"],
    plot_title="Training of MIL Classifier with Mean Aggregation",
    path_prefix=str(PROJECT_ROOT / "data/torch_playground_mean_aggregation_models/epoch_")
)