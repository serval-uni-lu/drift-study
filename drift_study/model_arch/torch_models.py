import torch.nn.functional as F
from mlc.models.torch_models import BaseModelTorch
from torch import nn


class ElectricityNet(nn.Module):
    def __init__(self, task: str, input_dim=12):
        super(ElectricityNet, self).__init__()
        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, 16)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend(
            [
                nn.Dropout(0.5),
                nn.Linear(16, 32),
                nn.Dropout(0.5),
            ]
        )

        # Output Layer
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all linear hidden layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)

        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x


class ElectricityModel(BaseModelTorch):
    def __init__(
        self,
        batch_size=64,
        epochs=10,
        early_stopping_rounds=2,
        learning_rate=0.001,
        **kwargs,
    ):
        name = "mlp_electricity"
        objective = "binary"
        super().__init__(
            name,
            objective,
            batch_size,
            epochs,
            early_stopping_rounds,
            learning_rate,
            **kwargs,
        )
        self.model = ElectricityNet(
            self.objective,
        )
        self.to_device()


class LcldNet(nn.Module):
    def __init__(self, task: str, input_dim=47):
        super(LcldNet, self).__init__()
        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, 32)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend(
            [
                nn.Dropout(0.5),
                nn.Linear(32, 32),
                nn.Dropout(0.5),
            ]
        )

        # Output Layer
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all linear hidden layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)

        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x


class LcldModel(BaseModelTorch):
    def __init__(
        self,
        batch_size=64,
        epochs=10,
        early_stopping_rounds=2,
        learning_rate=0.001,
        class_weight="balanced",
        **kwargs,
    ):
        name = "mlp_lcld"
        objective = "binary"
        super().__init__(
            name,
            objective,
            batch_size,
            epochs,
            early_stopping_rounds,
            learning_rate,
            class_weight=class_weight,
            **kwargs,
        )
        self.model = LcldNet(
            self.objective,
        )
        self.to_device()


models = [("mlp_electricity", ElectricityModel), ("mlp_lcld", LcldModel)]
