from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame

from mlc.models.model import Model
from mlc.models.torch_models import BaseModelTorch
from mlc.transformers.tab_scaler import TabScaler
from mlc.typing import NDFloat, NDNumber
from torch.utils.data import DataLoader

"""
    Custom implementation for the standard multi-layer perceptron
"""


class TORCHRLN(BaseModelTorch):
    def __init__(
        self,
        objective: str,
        x_metadata: pd.DataFrame,
        batch_size: int,
        epochs: int,
        early_stopping_rounds: int,
        num_classes: int,
        n_layers: int,
        hidden_dim: int,
        norm: int = 1,
        theta: float = -7.5,
        name: str = "torchrln",
        scaler: Optional[TabScaler] = None,
        **kwargs: Any,
    ) -> None:

        # Parameters
        self.objective = objective
        self.x_metadata = x_metadata
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.norm = norm
        self.theta = theta
        
        # Super call

        # Generate super call
        super().__init__(
            objective=objective,
            x_metadata=x_metadata,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_rounds=early_stopping_rounds,
            num_classes=num_classes,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            norm=norm,
            theta=theta,
            name=name,
            scaler=scaler,
            **kwargs,
        )
        # self.learning_rate = learning_rate
        # self.lr = self.learning_rate
        # Compatibility
        self.lr = self.learning_rate

        self.experiment = None
        self.scaler = scaler

        self.create_model()
        
    def create_model(self):
        self.num_features = (
            self.x_metadata.shape[0]
            if self.scaler is None
            else self.scaler.get_transformed_num_features()
        )

        self.model = MLP_ModelRLN(
            n_layers=self.n_layers,
            input_dim=self.num_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_classes,
            task=self.objective,
        )
        layer = self.model.layers[0]
        if self.scaler is not None:
            self.model = nn.Sequential(
                self.scaler.get_transorm_nn(), self.model
            )
            layer = self.model[1].layers[0]

        self.wrapper_model = self.model
        self.to_device()

        self.rln_callback = RLNCallback(
            layer,
            norm=self.norm,
            avg_reg=self.theta,
            learning_rate=self.lr,
        )

    def fit(
        self,
        x: NDFloat,
        y: NDNumber,
        x_val: Optional[NDFloat] = None,
        y_val: Optional[NDNumber] = None,
        custom_train_dataloader: DataLoader = None,
        custom_val_dataloader: DataLoader = None,
    ) -> None:
        x = np.array(x, dtype=float)

        if x_val is not None:
            x_val = np.array(x_val, dtype=float)

        else:
            x, x_val, y, y_val = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )

        x = torch.tensor(x).float()
        x_val = torch.tensor(x_val).float()
        self.rln_callback.on_train_begin()

        if self.scaler is None:
            self.scaler = TabScaler()
            self.scaler.fit(x, x_type=self.x_metadata["type"])
            self.create_model()
        
        if self.scaler is not None:
            previous_model = self.model
            self.model = self.model[1]
            x = self.scaler.transform(x)
            x_val = self.scaler.transform(x_val)

            out = super(TORCHRLN, self).fit(
                x,
                y,
                x_val,
                y_val,
                custom_train_dataloader=custom_train_dataloader,
                custom_val_dataloader=custom_val_dataloader,
                scaler=self.scaler,
            )
            self.model = previous_model
            return out

        return super(TORCHRLN, self).fit(x, y, x_val, y_val)

    def many_predict(self, x, n_pred) -> NDFloat:

        x = np.array(x, dtype=float)
        x = np.repeat(x, n_pred, axis=0)

        in_force_train = self.force_train
        self.force_train = True
        pred = self.predict_proba(x)
        self.force_train = in_force_train

        pred.reshape(-1, n_pred, pred.shape[1])

        return pred

    def predict_helper(
        self,
        x: Union[NDFloat, torch.Tensor, pd.DataFrame],
        load_all_gpu: bool = False,
    ) -> NDFloat:
        x = np.array(x, dtype=float)
        return super().predict_helper(x, load_all_gpu=load_all_gpu)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0005, 0.001
            ),
            "norm": trial.suggest_categorical("norm", [1, 2]),
            "theta": trial.suggest_int("theta", -12, -8),
        }
        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "hidden_dim": 10,
            "n_layers": 5,
            "learning_rate": 0.0005,
            "norm": 1,
            "theta": -10,
        }
        return params

    @staticmethod
    def get_name() -> str:
        return "torchrln"


class MLP_ModelRLN(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        task: str,
    ):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x


class RLNCallback(object):
    def __init__(
        self,
        layer: nn.Linear,
        norm: int = 1,
        avg_reg: float = -7.5,
        learning_rate: float = 6e5,
    ):
        """
        An implementation of Regularization Learning, described in https://arxiv.org/abs/1805.06440, as a Keras
        callback.
        :param layer: The Keras layer to which we apply regularization learning.
        :param norm: Norm of the regularization. Currently supports only l1 and l2 norms. Best results were obtained
        with l1 norm so far.
        :param avg_reg: The average regularization coefficient, Theta in the paper.
        :param learning_rate: The learning rate of the regularization coefficients, nu in the paper. Note that since we
        typically have many weights in the network, and we optimize the coefficients in the log scale, optimal learning
        rates tend to be large, with best results between 10^4-10^6.
        """
        super(RLNCallback, self).__init__()
        self._layer = layer

        self._prev_weights: Optional[torch.Tensor] = None
        self._weights: Optional[torch.Tensor] = None
        self._prev_regularization: Optional[torch.Tensor] = None
        self._avg_reg = avg_reg
        self._shape = torch.t(self._layer.weight).shape
        self._lambdas = DataFrame(np.ones(self._shape) * self._avg_reg)
        self._lr = learning_rate
        assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"
        self.norm = norm

    def on_train_begin(self, logs: Any = None) -> None:
        self._update_values()

    def on_batch_end(self, logs: Any = None) -> None:
        self._prev_weights = self._weights
        self._update_values()
        gradients = self._weights - self._prev_weights

        # Calculate the derivatives of the norms of the weights
        if self.norm == 1:
            norms_derivative = np.sign(self._weights)
        else:
            norms_derivative = self._weights * 2

        if self._prev_regularization is not None:
            # This is not the first batch, and we need to update the lambdas
            lambda_gradients = gradients.multiply(self._prev_regularization)
            self._lambdas -= self._lr * lambda_gradients

            # Project the lambdas onto the simplex \sum(lambdas) = Theta
            translation = self._avg_reg - self._lambdas.mean().mean()
            self._lambdas += translation

        # Clip extremely large lambda values to prevent overflow
        max_lambda_values = np.log(
            np.abs(self._weights / norms_derivative)
        ).fillna(np.inf)
        self._lambdas = self._lambdas.clip(upper=max_lambda_values)

        # Update the weights
        regularization = norms_derivative.multiply(np.exp(self._lambdas))
        self._weights -= regularization

        with torch.no_grad():
            self._layer.weight = nn.Parameter(
                torch.t(torch.Tensor(self._weights.values))
            )

        self._prev_regularization = regularization

    def _update_values(self) -> None:
        self._weights = DataFrame(torch.t(self._layer.weight.cpu().detach()))
        
    def save(self, path: str) -> None:
        self.scaler.save(f"{path}.scaler")
        return super().save(path)

    def load(self, path: str) -> None:
        self.scaler.load(f"{path}.scaler")
        return super().load(path)


models: List[Tuple[str, Type[Model]]] = [("torchrln", TORCHRLN)]
