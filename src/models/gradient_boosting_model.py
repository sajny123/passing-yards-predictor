from sklearn.ensemble import HistGradientBoostingRegressor

def train_hist_gb(X_train, y_train):
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model
