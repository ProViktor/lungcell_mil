import torch
import torch.nn as nn
from torch.nn import Module


class MIL_model(Module):
    def __init__(self, instance_encoder, bag_aggregator) -> None:
        super().__init__()
        self.instance_encoder = instance_encoder
        self.bag_aggregator = bag_aggregator


    def forward(self, bag):
        bag_encoding = self.instance_encoder(bag)
        bag_classification = self.bag_aggregator(bag_encoding)
        
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
        layers.append(nn.Dropout(p = 0.1))

        # hidden layers
        for i in range(n_hidden - 2):
            layers.append(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = 0.1))

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


class MaxAggergation(Module):
    def __init__(self, post_process = True, encoding_size = 40) -> None:
        super().__init__()
        if post_process:
            self.post = nn.Sequential(
                nn.Linear(in_features=encoding_size, out_features=2),
                #nn.ReLU(),
                #nn.Linear(in_features=10, out_features=10),
                #nn.ReLU(),
                #nn.Linear(in_features=10, out_features=2),
                #nn.ReLU(),
                nn.Softmax(dim=0),
            )
        else:
            self.post = nn.Identity
    def forward(self, bag_encoding):
        
        #max_aggregation = torch.max(bag_encoding, dim = 0)
        #max_elements, max_indices = bag_encoding.max(dim = 0)

        #mean aggregation
        max_elements = torch.mean(bag_encoding, dim = 0)
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
            #nn.ReLU(),
            #nn.Linear(in_features=10, out_features=10),
            #nn.ReLU(),
            #nn.Linear(in_features=10, out_features=2),
            #nn.ReLU(),
            nn.Softmax(dim=0),
        )

        for module in (self.w, self.V, self.decision):
            for layer in module:
                if hasattr(layer, "weight"):
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )

    def forward(self, bag_encoding):
        v_multiplied = self.tanh(self.V.forward(bag_encoding))

        alphas = torch.exp(self.w.forward(v_multiplied))
        alphas = alphas / torch.sum(alphas)
        

        bag_sum = torch.sum(alphas * bag_encoding, dim=0)
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
            #nn.ReLU(),
            #nn.Linear(in_features=10, out_features=10),
            #nn.ReLU(),
            #nn.Linear(in_features=10, out_features=2),
            #nn.ReLU(),
            nn.Softmax(dim=0),
        )


        for module in (self.w, self.V, self.U, self.decision):
            for layer in module:
                if hasattr(layer, "weight"):
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )

    def forward(self, bag_encoding):
        # bag = bag.to_dense()

        query = self.tanh(self.V.forward(bag_encoding))
        gate = self.sigmoid(self.U.forward(bag_encoding))

        alphas = torch.exp(self.w.forward(query*gate))
        alphas = alphas / torch.sum(alphas)

        bag_sum = torch.sum(alphas * bag_encoding, dim=0)
        decision = self.decision(bag_sum)
        return decision
