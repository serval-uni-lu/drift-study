import logging
import os

import configutils
import matplotlib.pyplot as plt
import seaborn as sns

from drift_study import detector_metrics_absolute

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run():
    config = configutils.get_config()

    df = detector_metrics_absolute.run()

    print(df)
    ax = sns.scatterplot(df, x="n_train", y="metric", hue="pareto_front")
    dataset_name = config.get("dataset").get("name")
    model_name = config.get("runs")[0].get("model").get("name")

    def plotlabel(xvar, yvar, label):
        ax.text(xvar + 0.002, yvar, label)

    df.apply(
        lambda x: plotlabel(x["n_train"], x["metric"], x["pareto_front"]),
        axis=1,
    )
    plt.show()
    plt.savefig(f"./reports/{dataset_name}/{model_name}_absolute.pdf")


if __name__ == "__main__":
    run()
