import pandas as pd
import os

url = "https://www.pro-football-reference.com/years/2024/passing.htm"

tables = pd.read_html(url)

df = tables[0]
df = df.drop(columns=["Rk", "Player", "Team", "Pos", "QBrec", "4QC", "GWD", "Awards"])

# print(df.columns)
# print(df.head())

os.makedirs("data", exist_ok=True) 
df.to_csv(f"data/qb_passing_2024.csv", index=False)