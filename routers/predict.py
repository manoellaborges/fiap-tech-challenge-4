from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib


router = APIRouter()

model = load_model('saved_models/stock_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

class StockData(BaseModel):
    historical_data: list

@router.post("/")
def predict(data: StockData):
    input_data = np.array(data.historical_data).reshape(-1, 1)
    scaled_data = scaler.transform(input_data).reshape(1, len(data.historical_data), 1)
    
    prediction = model.predict(scaled_data)
    unscaled_prediction = scaler.inverse_transform(prediction)[0][0]
    return {"predicted_price": float(unscaled_prediction)}

@router.get("/health")
def health_check():
    return {"status": "healthy"}
