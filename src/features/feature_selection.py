import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def select_top_features(X_train, y_train, X_test, k=10):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    feature_names = X_train.columns
    importances = rf.feature_importances_
    
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

    top_features = feat_imp["Feature"].head(k).tolist()
    return X_train[top_features], X_test[top_features], top_features
