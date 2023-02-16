from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def sample_date(
    x: pd.DataFrame,
    y: np.NDArray[Union[np.int_, np.float_]],
    t: Union[pd.Series, np.NDArray[np.int_]],
    minority_share: Optional[float],
) -> Tuple[
    pd.DataFrame,
    np.NDArray[Union[np.int_, np.float_]],
    Union[pd.Series, np.NDArray[np.int_]],
]:
    if minority_share is None:
        return x, y, t

    sampling_strategy = minority_share / (1 - minority_share)
    idx = np.arrange(len(x))
    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=42
    )
    idx_new, _ = sampler.fit_resample(idx, y)

    return x.iloc[idx_new], y[idx_new], t[idx_new]