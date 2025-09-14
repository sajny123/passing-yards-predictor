from sklearn.linear_model import Ridge

def train_ridge(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model