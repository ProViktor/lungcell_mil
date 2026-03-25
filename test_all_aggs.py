import torch
from mil.models import MeanAggergation, MaxAggergation, AttentionAggregation, GatedAttentionAggregation

bag_encoding = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0], [5.0, 0.0]])
batch_indices = torch.tensor([0, 0, 1, 1])

# Create instances
mean_agg = MeanAggergation(post_process=False)
max_agg = MaxAggergation(post_process=False)
att_agg = AttentionAggregation(encoding_size=2, attention_hidden_size=4)
gatt_agg = GatedAttentionAggregation(encoding_size=2, attention_hidden_size=4)

print("Mean:", mean_agg(bag_encoding, batch_indices))
print("Max:", max_agg(bag_encoding, batch_indices))
print("Att:", att_agg(bag_encoding, batch_indices))
print("Gatt:", gatt_agg(bag_encoding, batch_indices))
