import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
# from models.linear_model import train_linear
# from models.random_forest_model import train_random_forest
# from models.ridge_model import train_ridge
# from models.gradient_boosting_model import train_hist_gb
from features.feature_engineering import add_features
from models.factory import get_model
import argparse

def evaluate_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("R2 score: ", r2_score(y_test, y_pred))
    print ("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    return y_pred

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["linear", "ridge", "random_forest", "hist_gb"],)
args = parser.parse_args()

df = pd.read_csv("data/qb_passing_2024.csv")
df = add_features(df)

X = df.drop(columns=["Yds"])
y = df["Yds"]

n = 49
top_players = df.nlargest(n, "Yds")
X = top_players.drop(columns=["Yds"])
y = top_players["Yds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = get_model(args.model)
model.fit(X_train, y_train)


print(f"Model: {args.model}")
evaluate_data(model, X_test, y_test)


# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals)
# plt.axhline(0, color='red')
# plt.xlabel("Predicted yards")
# plt.ylabel("Residuals")
# plt.show()