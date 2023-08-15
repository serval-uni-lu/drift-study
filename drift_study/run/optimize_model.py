import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import configutils
import joblib
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from mlc.load_do_save import save_json
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric import Metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model import Model
from mlc.models.model_factory import get_model
from optuna import Trial
from optuna._callbacks import MaxTrialsCallback, RetryFailedTrialCallback
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.model_selection import TimeSeriesSplit

from drift_study.typing import NDFloat, NDNumber
from drift_study.utils.logging import configure_logger


class TimeOptimizer:
    def __init__(
        self,
        output_path: str,
        model_class: Type[Model],
        model_params: Dict[str, Any],
        metric_name: str,
        metric: Metric,
        n_iter: int,
        n_fold: int = 5,
        x_metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.metric_name = metric_name
        self.metric = metric
        self.n_iter = n_iter
        self.n_fold = n_fold
        self.x_metadata = x_metadata

        self.output_path = output_path
        self.logger = logging.getLogger(__name__)

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def _train_test_model(
        self,
        model_params,
        x_train: NDFloat,
        x_test: NDFloat,
        y_train: NDNumber,
        y_test: NDNumber,
    ):
        model = self.model_class(
            verbose=0, **model_params, x_metadata=self.x_metadata
        )

        # Train
        start = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start
        self.logger.info(f"Train time {train_time}")

        # Test
        start = time.time()
        metric_value = compute_metric(model, self.metric, x_test, y_test)
        test_time = time.time() - start
        self.logger.info(f"Test time {test_time}")
        metric_out = {
            "metrics": {self.metric_name: metric_value},
            "train_time": train_time,
            "test_time": test_time,
        }
        return metric_out

    def _objective_fold(
        self,
        run_idx: int,
        fold_idx,
        model_params,
        x_train: NDFloat,
        x_test: NDFloat,
        y_train: NDNumber,
        y_test: NDNumber,
    ) -> float:
        metric_out = self._train_test_model(
            model_params=model_params,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        save_json(
            metric_out,
            f"{self.output_path}/metrics_iter_{run_idx}_fold_{fold_idx}.json",
        )
        return metric_out["metrics"][self.metric_name]

    def _objective(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
    ) -> float:
        start = time.time()

        model_params = self.model_class.define_trial_parameters(
            trial, trial_params
        )
        model_params = {**self.model_params, **model_params}
        tscv = TimeSeriesSplit(n_splits=self.n_fold)
        metrics = [
            self._objective_fold(
                trial.number,
                fold_idx,
                model_params,
                x.iloc[train_index],
                x.iloc[test_index],
                y[train_index],
                y[test_index],
            )
            for fold_idx, (train_index, test_index) in enumerate(
                reversed(list(tscv.split(x)))
            )
        ]
        metric = float(np.mean(metrics))
        end = time.time() - start
        metric_out = {
            "metrics": {self.metric_name: metric},
            "iter_time": end,
        }
        save_json(
            model_params,
            f"{self.output_path}/model_params_iter_{trial.number}.json",
        )
        save_json(
            metric_out,
            f"{self.output_path}/metric_iter_{trial.number}.json",
        )

        return float(np.mean(metrics))

    def _get_sampler(self) -> Tuple[str, TPESampler]:
        sampler_path = f"{self.output_path}/optuna_sampler.joblib"
        if os.path.exists(sampler_path):
            sampler = joblib.load(sampler_path)
        else:
            sampler = TPESampler(
                n_startup_trials=5, seed=42, multivariate=True
            )
            joblib.dump(
                sampler,
                sampler_path,
            )
        return sampler_path, sampler

    def optimize(
        self,
        x_train: NDFloat,
        y_train: NDNumber,
        x_test: NDFloat,
        y_test: NDNumber,
    ):

        trial_params = {"n_features": x_train.shape[1]}

        failed_trial_callback = RetryFailedTrialCallback(max_retry=None)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.output_path}/optuna_study.db",
            heartbeat_interval=10,
            grace_period=20,
            failed_trial_callback=failed_trial_callback,
        )

        sampler_path, sampler = self._get_sampler()

        study = optuna.create_study(
            study_name="optuna_study",
            sampler=sampler,
            storage=storage,
            directions=["maximize"],
            load_if_exists=True,
        )

        n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))

        if n_completed == 0:
            default_params = self.model_class.get_default_params(trial_params)
            if default_params is not None:
                study.enqueue_trial(default_params)

        if n_completed < self.n_iter:
            study.optimize(
                lambda trial: self._objective(
                    trial, trial_params, x_train, y_train
                ),
                callbacks=[
                    MaxTrialsCallback(
                        self.n_iter,
                        states=(TrialState.COMPLETE,),
                    ),
                    lambda *_: joblib.dump(sampler, sampler_path),
                ],
            )

        save_json(
            study.best_trial.params, f"{self.output_path}/best_params.json"
        )
        metric_out = self._train_test_model(
            model_params=study.best_trial.params,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        save_json(
            metric_out,
            f"{self.output_path}/best_metrics.json",
        )


def run(config: Dict[str, Any]):

    dataset = get_dataset(config.get("dataset"))
    x, y = dataset.get_x_y()
    x_metadata = dataset.get_metadata(only_x=True)
    model_class = get_model(config.get("model"))
    model_params = config.get("model").get("params", {})
    print(model_params)
    test_start_idx = config.get("test_start_idx")
    x_train, y_train = (
        x.iloc[:test_start_idx],
        y[:test_start_idx],
    )
    x_test, y_test = (
        x.iloc[test_start_idx:],
        y[test_start_idx:],
    )
    output_path = (
        f"data/drift/{dataset.name}/{model_class.get_name()}/model_opt"
    )
    optimizer = TimeOptimizer(
        output_path=output_path,
        model_class=model_class,
        model_params=model_params,
        metric_name=config.get("metric").get("name"),
        metric=create_metric(config.get("metric")),
        n_iter=config.get("optimization_iter").get("model"),
        x_metadata=x_metadata,
    )

    optimizer.optimize(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)

    run(config=config)
