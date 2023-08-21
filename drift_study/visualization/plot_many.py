import pandas as pd
from drift_study.reports.graphics import scatterplot, lineplot


def filter_extreme(df, max_pareto_rank=1):
    filter_others = df["pareto_rank"] <= max_pareto_rank
    df = df[filter_others]
    return df


def run():
    path_delays = "./detector.html.csv"
    # path_no_delays = "./detector_no_delays.html.csv"
    path_half_delays = "./detector_half_delays.html.csv"
    path_twice_delays = "./detector_twice_delays.html.csv"

    df_delays = pd.read_csv(path_delays)
    # df_no_delays = pd.read_csv(path_no_delays)
    df_half_delays = pd.read_csv(path_half_delays)
    df_twice_delays = pd.read_csv(path_twice_delays)

    df_delays = filter_extreme(df_delays)
    # df_no_delays = filter_extreme(df_no_delays)
    df_half_delays = filter_extreme(df_half_delays)
    df_twice_delays = filter_extreme(df_twice_delays)
    
    
    df_half_delays["delay"] = "$\\delta_d / 2$"
    df_delays["delay"] = "$\\delta_d$"
    df_twice_delays["delay"] = "$2\\delta_d$"


    df_all = pd.concat([df_delays, df_half_delays, df_twice_delays])

    lineplot(
        df_all,
        "pareto.pdf",
        x="n_train",
        y="ml_metric",
        y_label="MCC",
        hue="delay",
        x_label="\\# Train",
        fig_size=(6, 4),
        legend_pos="best",
    )


if __name__ == "__main__":
    run()
