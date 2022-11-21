from typing import Union

import numpy as np
import pandas as pd
import torch
from autoattack import AutoAttack
from mlc.models.pipeline import Pipeline
from torch import nn


class OneToTwoLogits(nn.Module):
    @staticmethod
    def forward(x):
        x = torch.sigmoid(x)
        return torch.cat([1 - x, x], dim=-1)


class AdversarialDrift:
    def __init__(
        self,
        drift_detector,
        **kwargs,
    ):

        self.drift_detector = drift_detector
        self.model: Union[None, Pipeline] = None

    def fit(self, model: Pipeline, **kwargs):
        self.model = model
        self.drift_detector.fit(model=model, **kwargs)

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)

        x_t = self.model.transform(x)

        internal_model = self.model[-1].model
        internal_model = torch.nn.Sequential(internal_model, OneToTwoLogits())
        attack = AutoAttack(
            internal_model, norm="L2", eps=5.0, device="cpu", verbose=0
        )
        attack.attacks_to_run = [
            # "apgd-ce",
            # "apgd-dlr",
            "fab",
            # "square",
            # "apgd-t",
            # "fab-t",
        ]

        y = self.model.predict(x)
        x_t = torch.tensor(x_t).float()
        y = torch.tensor(y)
        # y = torch.reshape(y, (-1, 1))
        # y = torch.cat([1 - y, y], dim=1)
        x_adv = attack.run_standard_evaluation(x_t, y, bs=len(x_t))
        norm = np.linalg.norm(x_t - x_adv, axis=1)

        is_drift, is_warning, metrics = self.drift_detector.update(metric=norm)
        # metrics[f"adv_norm"] = norm

        return is_drift, is_warning, metrics

    @staticmethod
    def needs_label():
        return False
