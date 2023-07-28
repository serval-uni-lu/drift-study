from typing import Optional

from mlc.datasets.dataset import Dataset


def update_dataset_name(
    dataset: Dataset, minority_share: Optional[float]
) -> None:
    if minority_share is not None:
        dataset.name = f"{dataset.name}_{str(minority_share)}"
