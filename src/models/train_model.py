import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("data/qb_passing_2024.csv")

X = df.drop(columns=["Yds"])
y = df["Yds"]

n = 49
top_players = df.nlargest(n, "Yds")
X = top_players.drop(columns=["Yds"])
y = top_players["Yds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 score: ", r2_score(y_test, y_pred))
print ("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))