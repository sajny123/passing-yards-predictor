import joblib
import pandas as pd
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from features.feature_engineering import add_features


model_path = os.path.join(project_root, "models", "final_model.pkl")
model = joblib.load(model_path)

new_data = pd.DataFrame([{
    "G": 17,
    "GS": 17,
    "Cmp": 460,
    "Cmp%": 70.6,
    "Att": 652,
    "TD": 43,
    "1D": 253,
    "Y/G": 289.3,
    "Rate": 108.5
}])

new_data = add_features(new_data)
trained_features = model.feature_names_in_
new_data = new_data[trained_features]
prediction = model.predict(new_data)
print("Predicted Passing Yards: ", prediction[0])