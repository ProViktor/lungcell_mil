import json
from typing import Tuple, Dict, List
from pathlib import Path
import numpy as np
import torch
from torch.nn import Module
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from mil.CustomDataloader import CustomLoader
from mil import PROJECT_ROOT
from torch.nn import BCELoss

import wandb


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
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Performs one iteration of model parameters training.
    Balances the loss for both classes.

    Args:
        model (Module): model to train
        dataloader (CustomLoader): dataloader for training data
        criterion (_type_): loss function to minimize
        optimizer (_type_): optimizer which computes parameter updates
        device (_type_): device to compute on

    Returns:
        Tuple[float, torch.Tensor, torch.Tensor]: (epoch_loss, all_targets, all_outputs)
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

    all_targets = []
    all_outputs = []

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

        all_targets.append(batch_targets.detach().cpu())
        all_outputs.append(output.detach().cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, torch.cat(all_targets, dim=0), torch.cat(all_outputs, dim=0)


def evaluate(
    model: Module, dataloader: CustomLoader, criterion, device, sparse=True
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Evaluates model's loss function on provided dataset.
    Balancves the loss for both calsses.

    Args:
        model (Module): model to evaluate
        dataloader (CustomLoader): dataloader for validation data
        criterion (_type_): loss function to compute
        device (_type_): device to compute on

    Returns:
        Tuple[float, torch.Tensor, torch.Tensor]: (epoch_loss, all_targets, all_outputs)
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

    all_targets = []
    all_outputs = []

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

            all_targets.append(batch_targets.detach().cpu())
            all_outputs.append(output.detach().cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, torch.cat(all_targets, dim=0), torch.cat(all_outputs, dim=0)


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
    save_weights=False,
    device=None,
    sparse=True,
    run_params: Dict = None,
) -> Tuple[List[float], List[float], int]:
    """Performs a training run to evaluate model preformance. Saves the terained models
    of each epoch under `save_path_prefix{#epoch}.torch`.

    Args:
        model (Module): model to evaluate
        train_loader (CustomLoader): dataloader for training data
        validation_loader (CustomLoader): dataloader for validation loss
        criterion (_type_): loss function to minimize
        optimizer (_type_): optimizer which computes parameter updates
        num_epochs (int): number of epochs to train for
        save_path_prefix (str): Prefix path for saving model weights and summaries.
        ax (_type_, optional): Matplotlib axis to plot loss curves on. If None, no plots are shown. Defaults to None.
        plot_title (str, optional): Title for the plot.
        save_weights (bool, optional): If to save model weights in each step. Defaults to False.
        device (_type_, optional): Device to train model on. If None, cuda is selected if available. Defaults to None.
        sparse (bool, optional): If using sparse features. Defaults to True.
        run_params (Dict, optional): Dictionary of run parameters.

    Returns:
        Tuple[List[float], List[float], int]: (train_loss_history, validation_loss_history, epoch_min_loss)
    """
    wandb.login()
    wandb_settings = wandb.Settings(
        show_errors=True,  # Show error messages in the W&B App
        silent=True,      # Disable all W&B console output
        show_warnings=True # Show warning messages in the W&B App
    )
    run = wandb.init(project="lungcell_mil", dir=PROJECT_ROOT / "runs", settings=wandb_settings)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Local saving setup
    run_save_dir = PROJECT_ROOT / "runs" / "local_logs" / run.id
    run_save_dir.mkdir(parents=True, exist_ok=True)

    weights_save_dir = run_save_dir / "model_weights"
    weights_save_dir.mkdir(parents=True, exist_ok=True)

    # Create/update latest_run symlink
    latest_run_link = PROJECT_ROOT / "runs" / "latest_run"
    if latest_run_link.is_symlink() or latest_run_link.exists():
        latest_run_link.unlink()
    try:
        latest_run_link.symlink_to(run_save_dir, target_is_directory=True)
    except Exception:
        # Fallback if symlink fails (e.g. on Windows or restricted filesystem)
        pass

    run_save_path_prefix = weights_save_dir / Path(save_path_prefix).name

    if run_params:
        wandb.config.update(run_params)
        params_path = run_save_dir / "run_params.json"
        run_save_dir.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(run_params, f, indent=4)
        wandb.save(str(params_path))

    if hasattr(model, "get_details"):
        wandb.config.update(model.get_details())

    train_loss_history = []
    valid_loss_history = []

    train_acc_history = []
    train_bacc_history = []
    train_prec_history = []
    train_rec_history = []
    train_f1_history = []

    valid_acc_history = []
    valid_bacc_history = []
    valid_prec_history = []
    valid_rec_history = []
    valid_f1_history = []

    for epoch in range(num_epochs):
        train_loss, train_targets, train_outputs = train(
            model, train_loader, criterion, optimizer, device, sparse=sparse
        )
        valid_loss, valid_targets, valid_outputs = evaluate(
            model, validation_loader, criterion, device, sparse=sparse
        )

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        # wandb logging
        train_targets_cls = np.argmax(train_targets.numpy(), axis=1)
        train_preds_cls = np.argmax(train_outputs.numpy(), axis=1)

        valid_targets_cls = np.argmax(valid_targets.numpy(), axis=1)
        valid_preds_cls = np.argmax(valid_outputs.numpy(), axis=1)

        train_acc = accuracy_score(train_targets_cls, train_preds_cls)
        train_bal_acc = balanced_accuracy_score(train_targets_cls, train_preds_cls)
        train_prec = precision_score(
            train_targets_cls, train_preds_cls, average="macro", zero_division=0
        )
        train_rec = recall_score(
            train_targets_cls, train_preds_cls, average="macro", zero_division=0
        )
        train_f1 = f1_score(
            train_targets_cls, train_preds_cls, average="macro", zero_division=0
        )

        valid_acc = accuracy_score(valid_targets_cls, valid_preds_cls)
        valid_bal_acc = balanced_accuracy_score(valid_targets_cls, valid_preds_cls)
        valid_prec = precision_score(
            valid_targets_cls, valid_preds_cls, average="macro", zero_division=0
        )
        valid_rec = recall_score(
            valid_targets_cls, valid_preds_cls, average="macro", zero_division=0
        )
        valid_f1 = f1_score(
            valid_targets_cls, valid_preds_cls, average="macro", zero_division=0
        )

        train_acc_history.append(train_acc)
        train_bacc_history.append(train_bal_acc)
        train_prec_history.append(train_prec)
        train_rec_history.append(train_rec)
        train_f1_history.append(train_f1)

        valid_acc_history.append(valid_acc)
        valid_bacc_history.append(valid_bal_acc)
        valid_prec_history.append(valid_prec)
        valid_rec_history.append(valid_rec)
        valid_f1_history.append(valid_f1)

        wandb.log(
            {
                "train loss": train_loss,
                "train accuracy": train_acc,
                "train blanced accuracy": train_bal_acc,
                "train precision": train_prec,
                "train recall": train_rec,
                "train f1": train_f1,
                "validation loss": valid_loss,
                "validation accuracy": valid_acc,
                "val balanced accuracy": valid_bal_acc,
                "val precision": valid_prec,
                "val recall": valid_rec,
                "val f1": valid_f1,
                "epoch": epoch + 1,
            }
        )

        if save_weights:
            weights_save_dir.mkdir(parents=True, exist_ok=True)
            weight_path = f"{run_save_path_prefix}{epoch}.torch"
            torch.save(model, weight_path)

    # Calculate best epochs and values
    best_epoch_idx = {
        "val_loss": int(np.argmin(valid_loss_history)),
        "val_acc": int(np.argmax(valid_acc_history)),
        "val_bacc": int(np.argmax(valid_bacc_history)),
        "val_prec": int(np.argmax(valid_prec_history)),
        "val_rec": int(np.argmax(valid_rec_history)),
        "val_f1": int(np.argmax(valid_f1_history)),
    }

    best_values = {
        "val_loss": float(valid_loss_history[best_epoch_idx["val_loss"]]),
        "val_acc": float(valid_acc_history[best_epoch_idx["val_acc"]]),
        "val_bacc": float(valid_bacc_history[best_epoch_idx["val_bacc"]]),
        "val_prec": float(valid_prec_history[best_epoch_idx["val_prec"]]),
        "val_rec": float(valid_rec_history[best_epoch_idx["val_rec"]]),
        "val_f1": float(valid_f1_history[best_epoch_idx["val_f1"]]),
    }

    best_epochs = {k: v + 1 for k, v in best_epoch_idx.items()}

    # Log to wandb summary (flat keys for easier table filtering)
    for metric in best_epoch_idx.keys():
        wandb.summary[f"best_{metric}"] = best_values[metric]
        wandb.summary[f"best_{metric}_epoch"] = best_epochs[metric]

    # Map metrics to their best checkpoint file paths if saved
    best_checkpoints = {}
    if save_weights:
        for metric, idx in best_epoch_idx.items():
            best_checkpoints[metric] = str(
                (
                    weights_save_dir / f"{Path(save_path_prefix).name}{idx}.torch"
                ).absolute()
            )

    # Save local summary
    summary_data = {
        "best_epochs": best_epochs,
        "best_values": best_values,
        "best_checkpoints": best_checkpoints,
        "model_details": model.get_details() if hasattr(model, "get_details") else {},
        "run_id": run.id,
        "save_path_prefix": str(Path(save_path_prefix).absolute()),
    }

    run_save_dir.mkdir(parents=True, exist_ok=True)
    with open(run_save_dir / "run_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    history_array = np.array(valid_loss_history)
    m = np.min(history_array)
    epoch_min = np.argmin(valid_loss_history)
    if save_weights:
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

    wandb.finish()
    return train_loss_history, valid_loss_history, epoch_min
