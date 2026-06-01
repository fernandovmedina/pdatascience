from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI(title="Car Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load("model.pkl")

TRANSMISSION_MAP = {
    "Automatic": 0,
    "Manual": 1,
    "Other": 2,
    "Semi-Auto": 3,
}

FUEL_TYPE_MAP = {
    "Diesel": 0,
    "Electric": 1,
    "Hybrid": 2,
    "Other": 3,
    "Petrol": 4,
}

MANUFACTURER_MAP = {
    "BMW": 0,
    "audi": 1,
    "ford": 2,
    "hyundi": 3,
    "merc": 4,
    "skoda": 5,
    "toyota": 6,
    "vauxhall": 7,
    "volkswagen": 8,
}

SCALER_PARAMS = {
    "price":   {"mean": 16800.0, "std": 9500.0},
    "mileage": {"mean": 23000.0, "std": 20000.0},
    "tax":     {"mean": 120.0,   "std": 55.0},
    "mpg":     {"mean": 55.0,    "std": 15.0},
}


def scale(value: float, feature: str) -> float:
    mean = SCALER_PARAMS[feature]["mean"]
    std = SCALER_PARAMS[feature]["std"]
    return (value - mean) / std


class CarInput(BaseModel):
    price: float
    transmission: str
    mileage: float
    fuelType: str
    tax: float
    mpg: float
    engineSize: float
    Manufacturer: str
    age: int


class PredictionResponse(BaseModel):
    recommended: bool
    label: int


@app.get("/")
def root():
    return {"status": "Car Recommendation API is running"}


@app.get("/options")
def get_options():
    """Return valid categorical options for the frontend dropdowns."""
    return {
        "transmissions": list(TRANSMISSION_MAP.keys()),
        "fuelTypes": list(FUEL_TYPE_MAP.keys()),
        "manufacturers": list(MANUFACTURER_MAP.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CarInput):
    transmission_enc = TRANSMISSION_MAP.get(data.transmission, 1)
    fuel_enc = FUEL_TYPE_MAP.get(data.fuelType, 4)
    manufacturer_enc = MANUFACTURER_MAP.get(data.Manufacturer, 2)

    price_sc = scale(data.price, "price")
    mileage_sc = scale(data.mileage, "mileage")
    tax_sc = scale(data.tax, "tax")
    mpg_sc = scale(data.mpg, "mpg")

    features = np.array([[
        price_sc,
        transmission_enc,
        mileage_sc,
        fuel_enc,
        tax_sc,
        mpg_sc,
        data.engineSize,
        manufacturer_enc,
        data.age,
    ]])

    label = int(model.predict(features)[0])
    return PredictionResponse(recommended=bool(label), label=label)
