import glob

import pandas as pd

INPUT_DIR = "./notebooks/metrics/correlation_*.csv"


files = []
tables = [
    {
        "name": "label",
        "metric": "corr",
        "cols": ["variation", "distance"],
        "rows": ["dataset", "model"],
        "filters": {
            "adv_training": [False],
            "distance": ["distance_label", "distance_nolabel"],
        },
    },
    {
        "name": "adv_training",
        "metric": "corr",
        "cols": ["variation", "adv_training"],
        "rows": ["dataset", "model"],
        "filters": {
            "distance": ["distance_label"],
        },
    },
    {
        "name": "distance_confidence",
        "metric": "corr",
        "cols": ["distance"],
        "rows": ["dataset", "model"],
        "filters": {
            "adv_training": [False],
            "variation": [False],
            "distance": [
                "distance_label",
                "distance_nolabel",
                "confidence_label",
                "confidence_nolabel",
            ],
        },
    },
]


def create_table(df, table):
    df = df.copy()

    for filter_col, filter_values in table["filters"].items():
        df = df[df[filter_col].isin(filter_values)]

    df = df.pivot_table(
        index=table["rows"],
        columns=table["cols"],
        values=table["metric"],
    )
    return df


def load_metrics():
    # load all csv that match the input_dir regex and concatenate them
    df = pd.concat([pd.read_csv(f) for f in glob.glob(INPUT_DIR)])
    return df


def filter_files(df, files):
    pass


def run():
    df = load_metrics()
    print(df.shape)

    for table in tables:
        if len(files) == 0:
            p_table = create_table(df, table)
            path = f"./notebooks/tables/{table['name']}.xlsx"
            print(p_table)
            p_table.to_excel(path)
        else:
            for group_name, group in df.groupby(files):
                print(group_name)
                path = f"./notebooks/tables/{group_name}_{table['name']}.xlsx"
                p_table.to_excel(path)


if __name__ == "__main__":
    run()
