import torch
from mil.models import MaxAggergation

# Create sample bag encoding
bag_encoding = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0], [5.0, 0.0]])
batch_indices = torch.tensor([0, 0, 1, 1])

# Create MaxAggergation instance
agg = MaxAggergation(post_process=False)
result = agg(bag_encoding, batch_indices)

print("Result:", result)
print("Expected:", torch.tensor([[3.0, 2.0], [5.0, 4.0]]))
