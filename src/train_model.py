import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from models.linear_model import train_linear
from models.random_forest_model import train_random_forest
from models.ridge_model import train_ridge
from models.gradient_boosting_model import train_hist_gb

def evaluate_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("R2 score: ", r2_score(y_test, y_pred))
    print ("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    return y_pred

df = pd.read_csv("data/qb_passing_2024.csv")

X = df.drop(columns=["Yds"])
y = df["Yds"]

n = 49
top_players = df.nlargest(n, "Yds")
X = top_players.drop(columns=["Yds"])
y = top_players["Yds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = train_linear(X_train, y_train)
# model = train_random_forest(X_train, y_train)
# model = train_ridge(X_train, y_train)
model = train_hist_gb(X_train, y_train)

evaluate_data(model, X_test, y_test)





# model = LinearRegression()
# model.fit(X_train, y_train)

# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals)
# plt.axhline(0, color='red')
# plt.xlabel("Predicted yards")
# plt.ylabel("Residuals")
# plt.show()