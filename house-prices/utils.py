import pandas as pd
import numpy as np
from typing import List


def one_hot_encoding(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """receive a df and column names. One-hot encode the columns, merge to df and return result"""
    for col in cols:
        one_hot = pd.get_dummies(df[col])
        df = df.drop(col, axis=1)
        df = df.join(one_hot)
    return df
