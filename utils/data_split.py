import numpy as np
import pandas as pd

def temporal_train_test_split(
    df: pd.DataFrame, 
    date_col: str ="date", 
    test_size: float =0.2
    ):
    dates = df[date_col].values
    sorted_indices = np.argsort(dates)
    split_idx = int(len(dates) * (1 - test_size))
    split_date = dates[sorted_indices[split_idx]]

    mask = dates < split_date
    return df[mask], df[~mask]