import pandas as pd

detector_names = {
    "no_detection": "No detection",
    "periodic": "Periodic",
    "n_batch_tabular_alibi": "Statistical test",
    "n_batch_evidently": "Divergence",
    "data_evidently": "Divergence",
    "n_batch_pca_cd": "PCA-CD",
    "pca_cd": "PCA-CD",
    "adwin_class_error": "ADWIN Error rate",
    "adwin_proba_error": "ADWIN Proba error",
    "adwin_proba": "ADWIN Proba error",
    "adwin_class": "ADWIN Error rate",
    "ddm": "DDM",
    "eddm": "EDDM",
    "hddm_a": "HDDM-A",
    "hddm_a_error": "HDDM-A Error rate",
    "hddm_w_error": "HDDM-W Error rate",
    "hddm_w": "HDDM-W",
    "kswin_class_error": "KSWIN Error rate",
    "kswin_proba_error": "KSWIN Proba error",
    "page_hinkley_class_error": "Page-Hinkley Error rate",
    "page_hinkley_proba_error": "Page-Hinkley Proba error",
    "n_batch_rf_uncertainty_adwin": "Uncertainty",
    "rf_uncertainty": "Uncertainty",
    "n_batch_rf_uncertainty_adwin_y_scores": "Uncertainty",
    "n_batch_aries_adwin": "Aries ADWIN",
    "manual_rf_odav": "No detection",
    "aries_all": "Aries ADWIN",
    "tabular_alibi": "Statistical test",
    "ddm_error": "DDM",
    "eddm_error": "EDDM",
}

detector_types = {
    "baseline": "Baseline",
    "data": "Data",
    "error": "Error",
    "predictive": "Predictive",
}

delays_names = {
    "0": "No delay",
    "2W": "$\\delta_d / 2$",
    "4W": "$\\delta_d$",
    "8W": "$2 \\delta_d$",
}


def beautify_dataframe(
    df: pd.DataFrame, add_missing: bool = False
) -> pd.DataFrame:

    df["is_pareto"] = df["pareto_rank"] == 1
    df["size"] = df.groupby(["detector_name"])["detector_name"].transform(
        "size"
    )
    df["pareto_count"] = df.groupby(["detector_name"])["is_pareto"].transform(
        "sum"
    )

    def detector_name(x):
        if x in detector_names:
            return detector_names[x]
        for e in detector_names:
            if e in x:
                return detector_names[e]

        return x

    df["Detector"] = df["detector_name"].map(lambda x: detector_name(x))
    df["Type"] = df["detector_type"].map(lambda x: detector_types[x])

    df["Pareto / Total"] = (
        df[["pareto_count", "size"]].astype(str).agg(" / ".join, axis=1)
    )

    if add_missing:
        for i, e in enumerate(detector_names.keys()):
            if e not in df["detector_name"].to_list():
                df = df.append(
                    {
                        "detector_name": list(detector_names.keys())[i],
                        "Detector": list(detector_names.values())[i],
                        "Type": "TODO",
                        "Pareto / Total": "0 / 0",
                    },
                    ignore_index=True,
                )

    df["detector_order"] = df["detector_name"].map(
        lambda x: list(detector_names.keys()).index(x)
    )
    df = df.sort_values(by=["detector_order"], ignore_index=True)

    if "delays" in df:
        df["Delays"] = df["delays"].map(lambda x: delays_names[x])

    return df


name_to_type_dict = {
    "no_retrain": "Baseline",
    "divergence": "Data",
    "no_detection": "Baseline",
}


def name_to_type(name: str) -> str:
    name = name.split(("_iter_"))[0]
    if name.startswith("periodic"):
        return "Baseline"
    if name in name_to_type_dict:
        return name_to_type_dict[name]
    return "Other"
