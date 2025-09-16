from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

def get_model(name: str):
    if name == "linear":
        return LinearRegression()
    elif name == "ridge":
        return Ridge()
    elif name == "random_forest":
        return RandomForestRegressor()
    elif name == "hist_gb":
        return HistGradientBoostingRegressor()
    else:
        raise ValueError("Not valid model")