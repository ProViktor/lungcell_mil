import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from mil.CellsData import CellsData


def evaluate_accuracy(
    model: nn.Module, dataset: CellsData, bag_key: str, mode="Validation"
):
    y_true = []
    preds = []

    device = "cpu"
    model.to(device)
    for data_dict in dataset:
        bag = data_dict[bag_key].to(device)
        y = data_dict["y"].to(device)

        pred = model.forward(bag)
        preds.append(torch.round(pred).detach().numpy())
        y_true.append(y.detach().numpy())

    preds = np.array(preds)
    preds = preds.reshape(preds.shape[0], -1)
    preds = np.argmax(preds, axis=1)
    y_true = np.array(y_true)
    y_true = np.argmax(y_true, axis=1)

    print()
    acc = (preds == y_true).sum() / len(preds) * 100
    print(f"Accuracy: {acc:.2f}%")
    bal_acc = balanced_accuracy_score(y_true, preds) * 100
    print(f"Balanced Accuracy: {bal_acc:.2f}%")
    precision = precision_score(y_true, preds, average="macro", zero_division=0) * 100
    print(f"Precision: {precision:.2f}%")
    recall = recall_score(y_true, preds, average="macro", zero_division=0) * 100
    print(f"Recall: {recall:.2f}%")

    cm = confusion_matrix(y_true, preds)
    s = cm.sum(axis=1)
    d = np.diag(1 / s)
    cmn = d @ cm
    fig, ax = plt.subplots()

    sns.heatmap(cmn, ax=ax, vmin=0, vmax=1)
    title_string = f"Confusion Matrix  on the {mode} Set"
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title_string)
