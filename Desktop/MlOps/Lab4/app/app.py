import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

BASEDIR = os.path.dirname(os.path.abspath(__file__))
MODELPATH = os.path.join(BASEDIR, '..', 'artifacts', 'model.pkl')
model = joblib.load(MODELPATH)

class WineFeatures(BaseModel):
    fixedacidity: float
    volatileacidity: float
    citricacid: float
    residualsugar: float
    chlorides: float
    freesulfurdioxide: float
    totalsulfurdioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(features: WineFeatures):
    feature_array = np.array([[
        features.fixedacidity, features.volatileacidity, features.citricacid,
        features.residualsugar, features.chlorides, features.freesulfurdioxide,
        features.totalsulfurdioxide, features.density, features.pH,
        features.sulphates, features.alcohol
    ]])
    
    prediction = model.predict(feature_array)
    return {
        "name": "Urvashi Salonia",
        "rollno": "2022BCS0222",
        "winequality": int(round(prediction[0]))
    }

@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API"}
