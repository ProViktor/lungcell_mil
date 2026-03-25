import torch
import torch.nn as nn
from torch.nn import Module


class MIL_model(Module):
    def __init__(self, instance_encoder, bag_aggregator) -> None:
        super().__init__()
        self.instance_encoder = instance_encoder
        self.bag_aggregator = bag_aggregator

    def forward(self, bag, batch_indices=None):
        bag_encoding = self.instance_encoder(bag)
        is_single_bag = batch_indices is None
        if is_single_bag:
            batch_indices = torch.zeros(
                bag_encoding.size(0), dtype=torch.long, device=bag_encoding.device
            )
        bag_classification = self.bag_aggregator(bag_encoding, batch_indices)

        if is_single_bag:
            bag_classification = bag_classification.squeeze(0)

        return bag_classification


class MLP_encoder(Module):
    def __init__(
        self, n_hidden: int, hidden_size: int, input_size=2000, output_size=1
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.input_size = input_size

        layers = []
        # first layer
        layers.append(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        )
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.1))

        # hidden layers
        for i in range(n_hidden - 2):
            layers.append(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))

        # last layer - does not use ReLU
        layers.append(nn.Linear(in_features=self.hidden_size, out_features=output_size))

        self.instance_encoder = nn.Sequential(*layers)

        for layer in self.instance_encoder:
            if hasattr(layer, "weight"):
                torch.nn.init.kaiming_uniform_(
                    layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, bag):
        bag_encoding = self.instance_encoder(bag)
        return bag_encoding


class MeanAggergation(Module):
    def __init__(self, post_process=True, encoding_size=40) -> None:
        super().__init__()
        if post_process:
            self.post = nn.Sequential(
                nn.Linear(in_features=encoding_size, out_features=2),
                nn.Softmax(dim=1),
            )
        else:
            self.post = nn.Identity()

    def forward(self, bag_encoding, batch_indices):
        expanded_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))
        num_bags = batch_indices.max() + 1
        out = torch.zeros(
            num_bags,
            bag_encoding.size(1),
            dtype=bag_encoding.dtype,
            device=bag_encoding.device,
        )
        mean_elements = out.scatter_reduce_(
            dim=0,
            index=expanded_indices,
            src=bag_encoding,
            reduce="mean",
            include_self=False,
        )
        final = self.post(mean_elements)
        return final


class MaxAggergation(Module):
    def __init__(self, post_process=True, encoding_size=40) -> None:
        super().__init__()
        if post_process:
            self.post = nn.Sequential(
                nn.Linear(in_features=encoding_size, out_features=2),
                nn.Softmax(dim=1),
            )
        else:
            self.post = nn.Identity()

    def forward(self, bag_encoding, batch_indices):
        expanded_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))
        num_bags = batch_indices.max() + 1
        out = torch.zeros(
            num_bags,
            bag_encoding.size(1),
            dtype=bag_encoding.dtype,
            device=bag_encoding.device,
        )
        max_elements = out.scatter_reduce_(
            dim=0,
            index=expanded_indices,
            src=bag_encoding,
            reduce="amax",
            include_self=False,
        )
        final = self.post(max_elements)
        return final


class AttentionAggregation(Module):
    def __init__(self, encoding_size, attention_hidden_size) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.w = nn.Sequential(
            nn.Linear(in_features=attention_hidden_size, out_features=1)
        )
        self.V = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=attention_hidden_size)
        )

        self.decision = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=2),
            nn.Softmax(dim=1),
        )

        for module in (self.w, self.V, self.decision):
            for layer in module:
                if hasattr(layer, "weight"):
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )

    def forward(self, bag_encoding, batch_indices):
        v_multiplied = self.tanh(self.V.forward(bag_encoding))

        alphas = torch.exp(self.w.forward(v_multiplied))

        expanded_indices = batch_indices.unsqueeze(-1).expand(-1, alphas.size(1))
        num_bags = batch_indices.max() + 1
        sum_alphas = torch.zeros(
            num_bags, alphas.size(1), dtype=alphas.dtype, device=alphas.device
        ).scatter_reduce_(
            dim=0, index=expanded_indices, src=alphas, reduce="sum", include_self=False
        )
        alphas = alphas / sum_alphas[batch_indices]

        weighted_bag = alphas * bag_encoding
        bag_sum_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))
        bag_sum = torch.zeros(
            num_bags,
            bag_encoding.size(1),
            dtype=bag_encoding.dtype,
            device=bag_encoding.device,
        ).scatter_reduce_(
            dim=0,
            index=bag_sum_indices,
            src=weighted_bag,
            reduce="sum",
            include_self=False,
        )

        decision = self.decision(bag_sum)
        return decision


class GatedAttentionAggregation(Module):
    def __init__(self, encoding_size, attention_hidden_size) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.w = nn.Sequential(
            nn.Linear(in_features=attention_hidden_size, out_features=1)
        )
        self.V = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=attention_hidden_size)
        )

        self.U = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=attention_hidden_size)
        )

        self.decision = nn.Sequential(
            nn.Linear(in_features=encoding_size, out_features=2),
            nn.Softmax(dim=1),
        )

        for module in (self.w, self.V, self.U, self.decision):
            for layer in module:
                if hasattr(layer, "weight"):
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )

    def forward(self, bag_encoding, batch_indices):
        query = self.tanh(self.V.forward(bag_encoding))
        gate = self.sigmoid(self.U.forward(bag_encoding))

        alphas = torch.exp(self.w.forward(query * gate))

        expanded_indices = batch_indices.unsqueeze(-1).expand(-1, alphas.size(1))
        num_bags = batch_indices.max() + 1
        sum_alphas = torch.zeros(
            num_bags, alphas.size(1), dtype=alphas.dtype, device=alphas.device
        ).scatter_reduce_(
            dim=0, index=expanded_indices, src=alphas, reduce="sum", include_self=False
        )
        alphas = alphas / sum_alphas[batch_indices]

        weighted_bag = alphas * bag_encoding
        bag_sum_indices = batch_indices.unsqueeze(-1).expand(-1, bag_encoding.size(1))
        bag_sum = torch.zeros(
            num_bags,
            bag_encoding.size(1),
            dtype=bag_encoding.dtype,
            device=bag_encoding.device,
        ).scatter_reduce_(
            dim=0,
            index=bag_sum_indices,
            src=weighted_bag,
            reduce="sum",
            include_self=False,
        )

        decision = self.decision(bag_sum)
        return decision
