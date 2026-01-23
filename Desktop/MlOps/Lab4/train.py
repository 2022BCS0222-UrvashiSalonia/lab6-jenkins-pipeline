import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import json
import os

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

# Load wine quality dataset (standard UCI dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Prepare features and target
X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save metrics
metrics = {"r2": float(r2), "mse": float(mse)}
with open('artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save model
joblib.dump(model, 'artifacts/model.pkl')

print(f"Model trained - R2: {r2:.4f}, MSE: {mse:.4f}")
