import torch
from torch_scatter import scatter_mean, scatter_add

# Create sample bag encoding
bag_encoding = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0], [5.0, 0.0]])
batch_indices = torch.tensor([0, 0, 1, 1])

expanded_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))

# test mean
mean_scatter = scatter_mean(bag_encoding, expanded_indices, dim=0)
num_bags = batch_indices.max() + 1
mean_native = torch.zeros(num_bags, bag_encoding.size(1), dtype=bag_encoding.dtype, device=bag_encoding.device).scatter_reduce_(
    dim=0, index=expanded_indices, src=bag_encoding, reduce="mean", include_self=False
)

print("Mean scatter:", mean_scatter)
print("Mean native:", mean_native)

# test add
add_scatter = scatter_add(bag_encoding, expanded_indices, dim=0)
add_native = torch.zeros(num_bags, bag_encoding.size(1), dtype=bag_encoding.dtype, device=bag_encoding.device).scatter_reduce_(
    dim=0, index=expanded_indices, src=bag_encoding, reduce="sum", include_self=False
)

print("Add scatter:", add_scatter)
print("Add native:", add_native)
