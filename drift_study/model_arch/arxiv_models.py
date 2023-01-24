from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from mlc.models.torch_models import BaseModelTorch
from wildtime.networks.article import ArticleNetwork


class ArxivModel(BaseModelTorch):
    def __init__(
        self,
        batch_size: int = 512,
        epochs: int = 10,
        early_stopping_rounds: int = 2,
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> None:
        name = "arxiv_bert"
        objective = "classification"
        super().__init__(
            name,
            objective,
            batch_size,
            epochs,
            early_stopping_rounds,
            learning_rate,
            class_weight=None,
            # force_device="cpu",
            is_text=True,
            **kwargs,
        )

        self.model = ArticleNetwork(num_classes=172)
        for p in self.model.model[0].parameters():
            p.requires_grad = False
        self.to_device()

    def fit(
        self,
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        x_val: Optional[npt.NDArray[np.float_]] = None,
        y_val: Optional[
            Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
        ] = None,
        reset_weight: bool = False,
    ):
        x = x.reshape((x.shape[0], -1, 2))
        if x_val is not None:
            x_val = x.reshape((x_val.shape[0], -1, 2))
        super().fit(x, y, x_val, y_val, reset_weight)

    def predict_helper(
        self, x: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        x = x.reshape((x.shape[0], -1, 2))
        return super().predict_helper(x)


models = [
    ("arxiv_bert", ArxivModel),
]
