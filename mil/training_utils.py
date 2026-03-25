from typing import Tuple, Dict
import numpy as np
import torch
from torch.nn import Module
from mil.CustomDataloader import CustomLoader
from torch.nn import BCELoss


def get_sub_dataset(dataset, indices):
    otp_dataset = list()
    for i in range(len(dataset)):
        if i in indices:
            otp_dataset.append(dataset[i])
    return otp_dataset


def stratified_cv_split(dataset, k_cv, seed=0):
    pos_idx = list()
    neg_idx = list()
    for i in range(len(dataset)):
        if dataset[i]["y"][1].item() == 0.0:
            neg_idx.append(i)
        else:
            pos_idx.append(i)

    n_rep_pos = int(np.ceil(len(pos_idx) / k_cv))
    n_rep_neg = int(np.ceil(len(neg_idx) / k_cv))

    cv_base = list(np.arange(k_cv))

    pos_cv_base = cv_base * n_rep_pos
    pos_cv_base = pos_cv_base[: len(pos_idx)]

    neg_cv_base = cv_base * n_rep_neg
    neg_cv_base = pos_cv_base[: len(neg_idx)]

    rng = np.random.default_rng(seed=seed)

    rng.shuffle(neg_cv_base)
    rng.shuffle(pos_cv_base)

    pos_zip = list(zip(pos_cv_base, pos_idx))
    neg_zip = list(zip(neg_cv_base, neg_idx))

    all_indices = pos_zip + neg_zip

    otp_list = []
    for i in range(k_cv):
        k_sub = [el[1] for el in all_indices if el[0] == i]
        otp_list.append(k_sub)

    return otp_list


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return None


def get_weight(dataset) -> Dict[int, float]:
    """Computes class weights for balanced loss, normalized to sum up to 1.

    Args:
        dataset (CustomLoader.dataset): dataset for wich the weights should be computed

    Returns:
        _type_: _description_
    """
    n_pos = 0
    for el in dataset:
        if el["y"][1].item() == 1:
            n_pos += 1

    positive_ratio = n_pos / len(dataset)
    s = 1 / (1 - positive_ratio) + 1 / (positive_ratio)

    weight = dict()

    weight[0] = (1 / (1 - positive_ratio)) / s
    weight[1] = (1 / (positive_ratio)) / s

    return weight


def train(
    model: Module, dataloader: CustomLoader, criterion, optimizer, device, sparse=True
) -> float:
    """Performs one iteration of model parameters training.
    Balances the loss for both classes.

    Args:
        model (Module): model to train
        dataloader (CustomLoader): dataloader for training data
        criterion (_type_): loss function to minimize
        optimizer (_type_): optimizer which computes parameter updates
        device (_type_): device to compute on

    Returns:
        float: evaluated loss function
    """

    model.train()
    running_loss = 0.0
    weight = get_weight(dataloader.dataset)
    w = torch.tensor([weight[0], weight[1]]).to(device)
    criterion = BCELoss(weight=w)
    if sparse:
        bag_key = "bag"
    else:
        bag_key = "bag_embed"

    for batch in dataloader.batches():
        optimizer.zero_grad()

        bag_tensors = []
        targets = []
        batch_indices = []

        for idx, data_dict in enumerate(batch):
            bag = data_dict[bag_key].to(device)
            bag_tensors.append(bag)
            targets.append(data_dict["y"].to(device))
            batch_indices.append(
                torch.full((bag.shape[0],), idx, dtype=torch.long, device=device)
            )

        batch_bag = torch.cat(bag_tensors, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
        batch_targets = torch.stack(targets, dim=0)

        output = model.forward(batch_bag, batch_indices)

        batch_loss = criterion(output, batch_targets) * len(batch)

        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model: Module, dataloader: CustomLoader, criterion, device, sparse=True):
    """Evaluates model's loss function on provided dataset.
    Balancves the loss for both calsses.

    Args:
        model (Module): model to evaluate
        dataloader (CustomLoader): dataloader for validation data
        criterion (_type_): loss function to compute
        device (_type_): device to compute on

    Returns:
        float: evaluated loss function
    """
    model.eval()
    running_loss = 0.0
    weight = get_weight(dataloader.dataset)
    w = torch.tensor([weight[0], weight[1]]).to(device)
    criterion = BCELoss(weight=w)
    if sparse:
        bag_key = "bag"
    else:
        bag_key = "bag_embed"

    with torch.no_grad():
        for batch in dataloader.batches():
            bag_tensors = []
            targets = []
            batch_indices = []

            for idx, data_dict in enumerate(batch):
                bag = data_dict[bag_key].to(device)
                bag_tensors.append(bag)
                targets.append(data_dict["y"].to(device))
                batch_indices.append(
                    torch.full((bag.shape[0],), idx, dtype=torch.long, device=device)
                )

            batch_bag = torch.cat(bag_tensors, dim=0)
            batch_indices = torch.cat(batch_indices, dim=0)
            batch_targets = torch.stack(targets, dim=0)

            output = model.forward(batch_bag, batch_indices)
            batch_loss = criterion(output, batch_targets) * len(batch)

            running_loss += batch_loss.item()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def model_run(
    model: Module,
    train_loader: CustomLoader,
    validation_loader: CustomLoader,
    criterion,
    optimizer,
    num_epochs: int,
    save_path_prefix: str,
    ax=None,
    plot_title=None,
    verbose=True,
    device=None,
    sparse=True,
) -> Tuple[list, list, int]:
    """Performs a training run to evaluate model preformance. Saves the terained models
    of each epoch under `save_path_prefix{#epoch}.torch`.

    Args:
        model (Module): model to evaluate
        train_loader (CustomLoader): dataloader for training data
        valiodation_loader (CustomLoader): dataloader for validation loss
        criterion (_type_): loss function to minimize
        optimizer (_type_): optimizer which computes parameter updates
        num_epochs (int): number of epochs to train for
        ax (_type_, optional): Matplotlib axis to plot loss curves on. If None, no plots are shown. Defaults to None.
        verbose (bool, optional): If to print epoch losses. Defaults to True.
        device (_type_, optional): Device to train model on. If None, cuda is selected if available. Defaults to None.

    Returns:
        Tuple[list, list, int]: (train_loss_history, validation_loss_history, epoch_min)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_history = list()
    valid_loss_history = list()

    for epoch in range(num_epochs):
        train_loss = train(
            model, train_loader, criterion, optimizer, device, sparse=sparse
        )
        valid_loss = evaluate(
            model, validation_loader, criterion, device, sparse=sparse
        )

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        if verbose:
            from pathlib import Path

            Path(save_path_prefix).parent.mkdir(parents=True, exist_ok=True)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}"
            )
            torch.save(model, f"{save_path_prefix}{epoch}.torch")

    history_array = np.array(valid_loss_history)
    m = np.min(history_array)
    epoch_min = np.argmin(valid_loss_history)
    if verbose:
        print(f"Min valid loss: Epoch {epoch_min + 1}, {m:.4f}")

    if ax is not None:
        x = np.arange(num_epochs) + 1
        ax.plot(x, train_loss_history, label="Train Loss")
        ax.plot(x, valid_loss_history, label="Validation Loss")
        ax.set_title(plot_title)
        ax.set_xlabel("Number of Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.axvline(epoch_min + 1, color="r", linestyle=":")

    return train_loss_history, valid_loss_history, epoch_min
