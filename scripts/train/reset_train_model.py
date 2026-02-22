import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv("training_data_clean.csv")
features = joblib.load("model_features.joblib")

X = df[features]
y = df["target_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.05,
    max_iter=800,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "location_model.joblib")

print("Saved location_model.joblib")
print("Train R2:", model.score(X_train, y_train))
print("Test  R2:", model.score(X_test, y_test))
