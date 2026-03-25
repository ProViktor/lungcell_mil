import torch
import numpy as np
import warnings

# Suppress annoying anndata/scanpy warnings if any
warnings.filterwarnings("ignore")

from mil.models import MIL_model, MeanAggergation, MLP_encoder
from mil.CellsData import CellsData
from mil.CustomDataloader import CustomLoader


def main():
    print("Loading data...")
    dataset = CellsData(split="train")

    # We will just test with the first element using dataloader or raw indexing
    # We test on embedding to avoid sparse tensor complexity unless you use sparse inputs
    loader = CustomLoader(dataset, batchsize=4, shuffle=False)

    # Take one batch
    batches = loader.batches()
    first_batch = batches[0]

    print(f"Loaded {len(first_batch)} items in the first batch")

    # Determine the input size from the data
    input_size = first_batch[0]["bag"].shape[1]
    print(f"Data input size: {input_size}")

    print("\nInitializing model...")
    # Initialize a randomly initiated model
    encoder = MLP_encoder(
        n_hidden=3, hidden_size=40, input_size=input_size, output_size=40
    )
    aggregator = MeanAggergation(post_process=True, encoding_size=40)
    model = MIL_model(encoder, aggregator)

    model.eval()

    print("\n--- Test 1: Forward pass on a single unbatched instance ---")
    single_data = first_batch[0]
    single_bag = single_data["bag"]
    single_label = single_data["y"]

    print(f"Single bag shape: {single_bag.shape}")
    print(f"Single label shape: {single_label.shape}, true value: {single_label}")

    with torch.no_grad():
        single_pred = model(single_bag)

    print(f"Prediction shape: {single_pred.shape}, prediction value: {single_pred}")
    print(f"Correct label matched shape? {single_pred.shape == single_label.shape}")

    print("\n--- Test 2: Forward pass on a full batch ---")
    bag_tensors = []
    targets = []
    batch_indices = []

    for idx, data_dict in enumerate(first_batch):
        bag = data_dict["bag"]
        bag_tensors.append(bag)
        targets.append(data_dict["y"])
        batch_indices.append(torch.full((bag.shape[0],), idx, dtype=torch.long))

    batch_bag = torch.cat(bag_tensors, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)
    batch_targets = torch.stack(targets, dim=0)

    print(f"Batch bag shape (concatenated): {batch_bag.shape}")
    print(f"Batch indices shape: {batch_indices.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")

    with torch.no_grad():
        batch_pred = model(batch_bag, batch_indices)

    print(f"Batch prediction shape: {batch_pred.shape}")
    print(
        f"Correct batch label matched shape? {batch_pred.shape == batch_targets.shape}"
    )

    print("\nAll tests ran successfully!")


if __name__ == "__main__":
    main()
