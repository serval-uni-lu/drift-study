import pandas as pd
from drift_study.reports.graphics import scatterplot


def filter_extreme(df, max_n_train, max_pareto_rank=1):
    filter_baseline = df["Type"] == "Baseline"
    filter_others = (df["n_train"] > 1)  & (df["pareto_rank"] <= max_pareto_rank)
    df = df[(filter_baseline | filter_others) & (df["n_train"] < max_n_train)]
    return df

def run():
    path_delays = "./lcld/rq2.html.csv"
    path_no_delays ="./lcld/rq3.html.csv"
    
    df_delays = pd.read_csv(path_delays)
    df_no_delays = pd.read_csv(path_no_delays)
    max_n_train = 143
    
    df_delays = filter_extreme(df_delays, max_n_train)
    # df_delays["Type"] = df_delays["schedule_type"]
    df_no_delays = filter_extreme(df_no_delays, max_n_train)
    # df_no_delays["Type"] = df_no_delays["schedule_type"]
    
    max_y = max(df_delays["ml_metric"].max(), df_no_delays["ml_metric"].max())
    min_y = min(df_delays["ml_metric"].min(), df_no_delays["ml_metric"].min())
    
    delta = max_y - min_y
    
    max_y += delta * 0.1
    min_y -= delta * 0.1
        
    type_to_marker = {
        "Baseline": "x",
        "Data": "s",
        "Error": "^",
        "Predictive": "o",
    }

    for name, df in zip(["delays", "no_delays"], [df_delays, df_no_delays]):
        n_markers = df["Type"].unique()
        markers = [type_to_marker[e] for e in n_markers]
        scatterplot(
                df,
                f"{name}.pdf",
                x="n_train",
                y="ml_metric",
                y_label="MCC",
                hue="Type",
                x_label="\\# Train",
                fig_size=(6, 4),
                legend_pos="best",
                markers=markers,
                ylim=(min_y, max_y),
            )
    
if __name__ == "__main__":
    run()