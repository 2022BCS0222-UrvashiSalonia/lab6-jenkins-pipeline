from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import json


app = FastAPI()


with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('model/metrics.json', 'r') as f:
    metrics = json.load(f)


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def read_root():
    return {
        "message": "Wine Quality Prediction API - Lab 4",
        "student": "Urvashi Salonia",
        "roll_number": "2022BCS0222",
        "model": metrics.get("model", "Unknown"),
        "mse": metrics.get("mse", "N/A"),
        "r2_score": metrics.get("r2_score", "N/A")
    }


@app.post("/predict")
def predict(features: WineFeatures):
    feature_array = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])
    
    prediction = model.predict(feature_array)
    
    return {
        "name": "Urvashi Salonia",
        "roll_no": "2022BCS0222",
        "wine_quality": int(round(prediction[0]))
    }


@app.get("/predict")
def predict_get(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):
    feature_array = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])
    
    prediction = model.predict(feature_array)
    
    return {
        "name": "Urvashi Salonia",
        "roll_no": "2022BCS0222",
        "wine_quality": int(round(prediction[0]))
    }
