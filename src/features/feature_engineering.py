import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv("data/qb_passing_2024.csv")

    df = df.copy()

    df["TDPerGame"] = df["TD"] / df["G"]
    df["IntPerGame"] = df["Int"] / df["G"]
    df["SkPerGame"] = df["Sk"] / df["G"]

    return df.dropna()
