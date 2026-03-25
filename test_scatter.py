import torch
from torch_scatter import scatter_max

bag_encoding = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0], [5.0, 0.0]])
batch_indices = torch.tensor([0, 0, 1, 1])

expanded_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))
max_elements_scatter = scatter_max(bag_encoding, expanded_indices, dim=0)[0]

num_bags = batch_indices.max().item() + 1
max_elements_native = torch.zeros(num_bags, bag_encoding.size(1), dtype=bag_encoding.dtype, device=bag_encoding.device).scatter_reduce_(
    dim=0, index=expanded_indices, src=bag_encoding, reduce="amax", include_self=False
)

print("Scatter_max:")
print(max_elements_scatter)
print("Native scatter_reduce_:")
print(max_elements_native)

# test what happens with include_self=False when using zeros:
