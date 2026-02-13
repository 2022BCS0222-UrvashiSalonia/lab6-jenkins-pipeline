import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os


print("MLOps Lab 4 - Training Pipeline")
print("Name: Urvashi Salonia")
print("Roll Number: 2022BCS0222")
print("="*50)


print("\nLoading dataset")
df = pd.read_csv('dataset/winequality-red.csv', sep=';')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


X = df.drop('quality', axis=1)
y = df['quality']


print("\nPreprocessing")
print("No preprocessing applied")
print("All features used")


print("\nSplitting data")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


print("\nTraining model")
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully")


print("\nEvaluating model")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n" + "="*50)
print("RESULTS")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("="*50)


os.makedirs('model', exist_ok=True)


print("\nSaving model")
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model/model.pkl")


print("\nSaving metrics")
metrics = {
    "student": "Urvashi Salonia",
    "roll_number": "2022BCS0222",
    "model": "RandomForestRegressor",
    "mse": float(mse),
    "r2_score": float(r2)
}


with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to model/metrics.json")


print("\nTraining completed successfully!")
