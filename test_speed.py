import logging
import os
import time

from drift_study.utils.helpers import initialize

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run():
    config = {
        "dataset": {"name": "lcld_201317_ds_time"},
        "performance": {"n_jobs": 10},
        "experience": {"random_state": 42},
    }
    run_config = {
        "model": {
            "source": "drift_study.model_arch.tree_model",
            "name": "rf_lcld",
        }
    }
    dataset, model, x, y, t = initialize(config, run_config)
    start_time = time.time()
    logger.info("Start")
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    model[0].fit(x.iloc[:100000], y[:100000])
    x_l = model[0].transform(x.iloc[:100000])
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    model[1].fit(x_l, y[:100000])
    logger.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    run()
